"""FaceSpoof — real-time face-tracked PNG overlay rendered to a virtual camera.

Pipeline:
    physical camera -> OpenCV capture -> MediaPipe Face Mesh (468 landmarks)
    -> alpha-blended PNG overlay positioned/rotated/scaled to the face
    -> pyvirtualcam sink (OBS Virtual Camera or Unity Capture)

Windows only. Requires Python 3.10-3.12 64-bit.
"""

import math
import os
import sys
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# MediaPipe is optional at runtime: if it fails to load we degrade to
# passthrough mode (raw feed, no overlay) instead of crashing.
try:
    import mediapipe as _mp
except Exception:
    _mp = None

# pyvirtualcam: availability is probed by opening a short-lived camera,
# not by version-specific helper functions (modern versions expose none).
try:
    import pyvirtualcam
except Exception:
    pyvirtualcam = None

UNITY_CAPTURE_URL = "https://github.com/schellingb/UnityCapture"
OBS_DOWNLOAD_URL = "https://obsproject.com/"

TARGET_FPS = 30
PROBE_INDICES_MAX = 9
CAMERA_READ_FAILURE_LIMIT = 60  # consecutive failed reads tolerated (~1-3 s)

MESH_MAX_WIDTH = 640       # face mesh runs on a downscaled frame (landmarks
                           # are normalized, so coordinates are unchanged)
FACE_LOST_RESET_FRAMES = 15  # frames without a face before filters reset
FEATHER_FRACTION = 0.06    # alpha edge feather as fraction of overlay size
MAX_FACES = 3              # simultaneous faces tracked and overlaid
HAAR_PAD = 1.3             # Haar boxes are tight; pad them like the mesh box
ADAPTIVE_RETRY_FRAMES = 20 # lost-face interval for full-resolution retry

# ---------------------------------------------------------------- styling ---

STYLESHEET = """
QWidget {
    background-color: #0D0D0D;
    font-family: "Segoe UI";
    font-size: 10pt;
    color: #B0B0B0;
}
QLabel {
    color: #B0B0B0;
    font-size: 10pt;
}
QComboBox {
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    color: #E0E0E0;
    font-size: 9pt;
    padding: 4px 8px;
    border-radius: 0px;
}
QComboBox:hover {
    background-color: #252525;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #555555;
    width: 0px;
    height: 0px;
}
QComboBox QAbstractItemView {
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    color: #E0E0E0;
    selection-background-color: #252525;
    selection-color: #E0E0E0;
    border-radius: 0px;
}
QPushButton {
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    color: #E0E0E0;
    font-size: 9pt;
    border-radius: 0px;
}
QPushButton:hover {
    background-color: #252525;
}
QPushButton:disabled {
    background-color: #141414;
    border: 1px solid #1F1F1F;
    color: #555555;
}
QPushButton#startButton {
    background-color: #1B3A1B;
    border: 1px solid #2A5A2A;
    color: #7ACC7A;
}
QPushButton#startButton:hover {
    background-color: #254A25;
}
QPushButton#startButton:disabled {
    background-color: #122412;
    border: 1px solid #1C301C;
    color: #4A6A4A;
}
QPushButton#stopButton {
    background-color: #3A1B1B;
    border: 1px solid #5A2A2A;
    color: #CC7A7A;
}
QPushButton#stopButton:hover {
    background-color: #4A2525;
}
QPushButton#stopButton:disabled {
    background-color: #241212;
    border: 1px solid #301C1C;
    color: #6A4A4A;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #1A1A1A;
    border-radius: 0px;
}
QSlider::handle:horizontal {
    width: 10px;
    height: 16px;
    margin: -6px 0;
    background: #4A4A4A;
    border-radius: 0px;
}
QSlider::handle:horizontal:hover {
    background: #5A5A5A;
}
QSlider::sub-page:horizontal {
    background: #1A1A1A;
}
QCheckBox {
    color: #B0B0B0;
    font-size: 9pt;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #2A2A2A;
    background: #1A1A1A;
}
QCheckBox::indicator:hover {
    border-color: #555555;
}
QCheckBox::indicator:checked {
    background: #7ACC7A;
    border-color: #2A5A2A;
}
"""

STATUS_COLORS = {
    "idle": "#666666",
    "running": "#7ACC7A",
    "error": "#CC7A7A",
}


# -------------------------------------------------------------- smoothing ---


class OneEuroFilter:
    """1-euro adaptive low-pass filter (Casiez et al. 2012).

    The same technique Google's FaceLandmarker applies to landmarks in
    streaming mode: heavy smoothing when the face is still (no jitter),
    nearly no lag during fast motion (adaptive cutoff rises with speed).
    """

    def __init__(self, x0, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = float(x)
            return self.x_prev
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.t_prev = None
        self.dx_prev = 0.0


# ------------------------------------------------------------ camera probe --


def list_camera_names():
    """Friendly names of DirectShow video input devices, in index order.

    Uses pygrabber (comtypes) so the chooser can show real device names;
    the returned order matches OpenCV's CAP_DSHOW index order. Returns []
    when pygrabber is unavailable or COM fails - callers fall back to
    "Camera {index}" labels and blind probing.
    """
    try:
        import comtypes
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return []
    try:
        comtypes.CoInitialize()
    except Exception:
        pass
    try:
        return [str(name) for name in FilterGraph().get_input_devices()]
    except Exception:
        return []


def probe_camera(index: int):
    """Return (ok, detail) for a camera index.

    A camera counts as usable only if it opens AND delivers at least one
    real frame. Tries DirectShow first, then Media Foundation, since some
    cameras only work with one of them.
    """
    detail = []
    for backend, backend_name in ((cv2.CAP_DSHOW, "DirectShow"),
                                  (cv2.CAP_MSMF, "MediaFoundation")):
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                detail.append(f"{backend_name}: open failed")
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            got_frame = False
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    got_frame = True
                    break
            if got_frame:
                return True, None
            detail.append(f"{backend_name}: opened but delivered no frames")
        except Exception as exc:
            detail.append(f"{backend_name}: {exc}")
        finally:
            if cap is not None:
                cap.release()
    return False, "; ".join(detail)


class CameraScanner(QThread):
    """Background scan so the UI never freezes while probing cameras.

    Only cameras that pass probe_camera() are reported - the chooser lists
    active, frame-delivering devices exclusively.
    """

    scanned = Signal(list)  # list of {"index", "name", "detail"}

    def __init__(self):
        super().__init__()
        self._abort = False

    def abort(self):
        """Stop after the current probe; cannot interrupt a blocking read."""
        self._abort = True

    def run(self):
        names = list_camera_names()
        if names:
            indices = range(min(len(names), PROBE_INDICES_MAX + 1))
        else:
            indices = range(PROBE_INDICES_MAX + 1)
        ready = []
        for index in indices:
            if self._abort:
                break
            ok, detail = probe_camera(index)
            if ok:
                ready.append({
                    "index": index,
                    "name": names[index] if index < len(names) else "",
                    "detail": detail or "",
                })
        if not self._abort:
            self.scanned.emit(ready)


# ------------------------------------------------------------- processing ---


FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/"
                       "face_landmarker/face_landmarker/float16/1/"
                       "face_landmarker.task")


def _ensure_landmarker_model() -> str:
    """Fetch the Tasks FaceLandmarker model once into a per-user cache dir
    (zero manual steps; raises on failure so callers fall back to Haar)."""
    cache_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "facespoof")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, "face_landmarker.task")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    tmp = dest + ".part"
    with urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=30) as resp, \
            open(tmp, "wb") as out:
        out.write(resp.read())
    os.replace(tmp, dest)
    return dest


class _LandmarkSet:
    __slots__ = ("landmark",)

    def __init__(self, landmark):
        self.landmark = landmark


class _MeshResult:
    __slots__ = ("multi_face_landmarks",)

    def __init__(self, multi_face_landmarks):
        self.multi_face_landmarks = multi_face_landmarks


class _TasksMeshAdapter:
    """mediapipe >= 0.10.35 removed the legacy Solutions API entirely
    (there is no mediapipe.solutions anymore); adapt the Tasks
    FaceLandmarker to the .process() interface the engine expects."""

    def __init__(self, static_mode: bool):
        vision = _mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=vision.BaseOptions(
                model_asset_path=_ensure_landmarker_model()),
            running_mode=(vision.RunningMode.IMAGE if static_mode
                          else vision.RunningMode.VIDEO),
            num_faces=MAX_FACES,
            min_face_detection_confidence=0.4 if static_mode else 0.5,
            min_face_presence_confidence=0.4 if static_mode else 0.5)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._static = static_mode
        self._ts_ms = 0

    def process(self, rgb):
        image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        if self._static:
            result = self._landmarker.detect(image)
        else:
            self._ts_ms += 33
            result = self._landmarker.detect_for_video(image, self._ts_ms)
        sets = [_LandmarkSet(l)
                for l in (getattr(result, "face_landmarks", None) or [])]
        return _MeshResult(sets)

    def close(self):
        self._landmarker.close()


class CaptureThread(QThread):
    """Captures frames, composites the overlay, pushes to the virtual camera.

    The virtual camera sink is instantiated inside run() so all driver
    interaction stays on this thread. Overlay and slider values are handed
    over through lock-guarded attributes so the UI thread never touches Qt
    widgets from the worker side.
    """

    frame_ready = Signal(object)       # RGB uint8 ndarray (H, W, 3)
    status_changed = Signal(str, str)  # text, kind in {"running", "error", "error-soft"}

    def __init__(self, camera_index: int, width: int, height: int, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._running = True

        self._overlay_lock = threading.Lock()
        self._overlay_rgb = None        # uint8 (H, W, 3)
        self._overlay_alpha = None      # float32 (H, W) in [0, 1]
        self._overlay_present = False
        self._overlay_luma = 0.0        # alpha-weighted mean luma at load

        self._opacity = 0.85
        self._scale = 1.0
        self._skin_match = False
        self._passthrough = False
        self._face_mesh = None

        # Per-face tracking slots: one filter + warp-cache set each, so
        # multiple faces overlay independently.
        self._slots = [self._new_slot() for _ in range(MAX_FACES)]
        self._haar = None
        self._show_tracking = False
        self._frame_idx = 0
        self._lost_frames = 0
        self._gain = 1.0
        self._f_gain = None

    # ---- controls called from the UI thread ----

    def set_overlay(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        with self._overlay_lock:
            self._overlay_rgb = rgb
            self._overlay_alpha = alpha
            self._overlay_present = True
            # Alpha-weighted mean luma, the reference for brightness matching:
            # only visible pixels count.
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            luma = rgb.astype(np.float32) @ weights
            denom = float(alpha.sum())
            self._overlay_luma = (float((luma * alpha).sum()) / denom
                                  if denom > 1e-6 else float(luma.mean()))
            self._gain = 1.0
            self._f_gain = OneEuroFilter(1.0, min_cutoff=0.6, beta=0.2)

    def clear_overlay(self) -> None:
        with self._overlay_lock:
            self._overlay_rgb = None
            self._overlay_alpha = None
            self._overlay_present = False
        self._gain = 1.0

    def set_skin_match(self, enabled: bool) -> None:
        self._skin_match = bool(enabled)

    def set_show_tracking(self, enabled: bool) -> None:
        self._show_tracking = bool(enabled)

    @staticmethod
    def _new_slot():
        return {"f_cx": None, "f_cy": None, "f_tw": None, "f_th": None,
                "f_ang": None, "f_wf": None,
                "cache_key": None, "cache_ov": None, "cache_al": None}

    def set_opacity(self, value: int) -> None:
        self._opacity = value / 100.0

    def set_scale(self, value: int) -> None:
        self._scale = value / 100.0

    def stop(self) -> None:
        self._running = False

    # ---- worker ----

    def _open_camera(self):
        """Open with DirectShow, fall back to Media Foundation.

        Some cameras only cooperate with one of the two backends. MJPG is
        requested before the resolution so 1080p runs at full frame rate on
        cameras that default to YUY2.
        """
        errors = []
        for backend, backend_name in ((cv2.CAP_DSHOW, "DirectShow"),
                                      (cv2.CAP_MSMF, "MediaFoundation")):
            cap = cv2.VideoCapture(self._camera_index, backend)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{backend_name}: open failed")
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, None
            cap.release()
            errors.append(f"{backend_name}: opened but delivered no frames")
        return None, "; ".join(errors)

    def run(self) -> None:
        cap, open_error = self._open_camera()
        if cap is None:
            self.status_changed.emit(
                f"Error: No device at index {self._camera_index}", "error")
            return

        if _mp is not None:
            try:
                if hasattr(_mp, "solutions"):
                    self._face_mesh = _mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=MAX_FACES,
                        refine_landmarks=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                else:
                    # mediapipe >= 0.10.35: legacy API gone, use Tasks.
                    self._face_mesh = _TasksMeshAdapter(static_mode=False)
            except Exception:
                self._face_mesh = None
                self._passthrough = True
                self.status_changed.emit("Error: Face mesh failed", "error-soft")
        else:
            self._passthrough = True
            self.status_changed.emit("Error: Face mesh failed", "error-soft")

        try:
            vcam = pyvirtualcam.Camera(
                self._width, self._height, fps=TARGET_FPS,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
        except Exception:
            self.status_changed.emit("Error: No virtual camera driver", "error")
            cap.release()
            return

        self.status_changed.emit("Running", "running")

        failures = 0
        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Single failed reads happen on healthy cameras; only give
                # up after a sustained run of them.
                failures += 1
                if failures > CAMERA_READ_FAILURE_LIMIT:
                    self.status_changed.emit(
                        f"Error: No device at index {self._camera_index}", "error")
                    break
                self.msleep(15)
                continue
            failures = 0
            self._frame_idx += 1

            # Drivers sometimes ignore the requested resolution; normalize so
            # the virtual camera dimensions always match the pushed frames.
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height),
                                   interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not self._passthrough and self._overlay_present:
                # MediaPipe landmarks are normalized, so detection on a
                # downscaled copy yields the same coordinates at a fraction
                # of the CPU cost on 720p/1080p input.
                det_scale = 1.0
                proc = rgb
                if rgb.shape[1] > MESH_MAX_WIDTH:
                    det_scale = MESH_MAX_WIDTH / rgb.shape[1]
                    proc = cv2.resize(
                        rgb, (MESH_MAX_WIDTH,
                              max(2, int(round(rgb.shape[0] * det_scale)))),
                        interpolation=cv2.INTER_AREA)
                t = self._frame_idx / TARGET_FPS
                faces = self._detect_faces(proc, det_scale)
                if not faces and self._lost_frames > 0 \
                        and self._lost_frames % ADAPTIVE_RETRY_FRAMES == 0 \
                        and det_scale < 1.0:
                    # Lost at reduced resolution: periodically retry at full
                    # resolution before giving up on the face.
                    faces = self._detect_faces(rgb, 1.0)
                    det_scale = 1.0
                if faces:
                    self._lost_frames = 0
                    self._composite_all(rgb, faces, t, det_scale)
                else:
                    self._lost_frames += 1
                    if self._lost_frames == FACE_LOST_RESET_FRAMES:
                        self._reset_tracking()
                # No face: emit the untouched frame.
                if self._show_tracking:
                    self._draw_hud(rgb, faces, det_scale)

            self.frame_ready.emit(rgb)
            vcam.send(rgb)
            vcam.sleep_until_next_frame()

        cap.release()
        vcam.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    # ---- compositing ----

    def _reset_tracking(self):
        self._slots = [self._new_slot() for _ in range(MAX_FACES)]

    def _detect_faces(self, proc, det_scale):
        """Tiered detection: MediaPipe mesh first, OpenCV Haar fallback.

        The Haar tier guarantees face detection even when MediaPipe is
        unavailable or refuses the frame. Faces are sorted largest-first
        and capped at MAX_FACES. Haar boxes are converted to full-frame
        pixels here; mesh landmark lists stay normalized.
        """
        faces = []
        if self._face_mesh is not None:
            try:
                result = self._face_mesh.process(proc)
            except Exception:
                result = None
            if result is not None and result.multi_face_landmarks:
                def area(lms):
                    xs = [lm.x for lm in lms]
                    ys = [lm.y for lm in lms]
                    return (max(xs) - min(xs)) * (max(ys) - min(ys))
                best = sorted(result.multi_face_landmarks,
                              key=lambda f: area(f.landmark), reverse=True)
                faces = [("mesh", f.landmark) for f in best[:MAX_FACES]]
        if not faces:
            # Haar fallback. Guarded: OpenCV 5.x wheels removed
            # CascadeClassifier and the bundled cascade XMLs, and the tier
            # must degrade to no-fallback instead of crashing the loop.
            if self._haar is None and hasattr(cv2, "CascadeClassifier"):
                cascade_path = os.path.join(
                    str(cv2.data.haarcascades),
                    "haarcascade_frontalface_default.xml")
                if os.path.exists(cascade_path):
                    self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar is not None:
                gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
                boxes = self._haar.detectMultiScale(
                    gray, 1.1, 5, minSize=(48, 48))
                if boxes is not None:
                    boxes = sorted(boxes, key=lambda b: -int(b[2]) * int(b[3]))
                    faces = [("haar",
                              (int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                             for b in boxes[:MAX_FACES]]
        if det_scale < 1.0:
            faces = [(kind, (data if kind == "mesh" else
                             tuple(v / det_scale for v in data)))
                     for kind, data in faces]
        return faces

    def _draw_hud(self, frame_rgb, faces, det_scale):
        """Debug overlay: green box + face index per detection."""
        frame_h, frame_w = frame_rgb.shape[:2]
        for i, (kind, data) in enumerate(faces):
            if kind == "mesh":
                xs = [lm.x * frame_w for lm in data]
                ys = [lm.y * frame_h for lm in data]
                x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            else:
                x0, y0, x1, y1 = int(data[0]), int(data[1]), \
                    int(data[0] + data[2]), int(data[1] + data[3])
            cv2.rectangle(frame_rgb, (x0, y0), (x1, y1), (60, 220, 60), 2)
            cv2.putText(frame_rgb, f"#{i + 1}", (x0, max(14, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

    def _face_geometry(self, landmarks, frame_w, frame_h):
        """Raw (unfiltered) overlay placement for one frame.

        Returns None if the face is too small to matter. Placement is built
        from stable anchors rather than raw bounding-box extremes:
        - rotation from the eye line (33/133 vs 362/263),
        - horizontal center pulled toward the nose tip (1) so the overlay
          follows head yaw,
        - width foreshortened as the head turns (cheeks 234/454 vs nose),
        - sizes from the landmark bounding box with the spec's 1.2 padding.
        """
        count = len(landmarks)
        xs = np.fromiter((lm.x for lm in landmarks), dtype=np.float64, count=count)
        ys = np.fromiter((lm.y for lm in landmarks), dtype=np.float64, count=count)
        x_min, x_max = xs.min() * frame_w, xs.max() * frame_w
        y_min, y_max = ys.min() * frame_h, ys.max() * frame_h
        face_w = x_max - x_min
        face_h = y_max - y_min
        if face_w < 8 or face_h < 8:
            return None

        lx = (landmarks[33].x + landmarks[133].x) * 0.5 * frame_w
        ly = (landmarks[33].y + landmarks[133].y) * 0.5 * frame_h
        rx = (landmarks[362].x + landmarks[263].x) * 0.5 * frame_w
        ry = (landmarks[362].y + landmarks[263].y) * 0.5 * frame_h
        angle = math.degrees(math.atan2(ry - ly, rx - lx))

        nose_x = landmarks[1].x * frame_w
        cheek_l = landmarks[234].x * frame_w
        cheek_r = landmarks[454].x * frame_w
        denom = cheek_r - cheek_l
        if abs(denom) >= 1.0:
            yaw_ratio = min(max((nose_x - cheek_l) / denom, 0.0), 1.0)
        else:
            yaw_ratio = 0.5
        # 0.5 = facing camera -> factor 1.0; turning shrinks the visible face
        # width, so the overlay must narrow with it (capped at 70 degrees
        # worth of narrowing so a full profile still looks sane).
        width_factor = 0.75 + 0.25 * math.cos(
            math.radians(min(abs(yaw_ratio - 0.5) * 155.0, 70.0)))

        cx = (x_min + x_max) * 0.5 + 0.3 * (nose_x - (x_min + x_max) * 0.5)
        cy = (y_min + y_max) * 0.5
        tw = face_w * self._scale * 1.2
        th = face_h * self._scale * 1.2
        return {"cx": cx, "cy": cy, "tw": tw, "th": th,
                "angle": angle, "wf": width_factor}

    def _init_filters(self, geom, slot):
        slot["f_cx"] = OneEuroFilter(geom["cx"], min_cutoff=1.2, beta=0.02)
        slot["f_cy"] = OneEuroFilter(geom["cy"], min_cutoff=1.2, beta=0.02)
        slot["f_tw"] = OneEuroFilter(geom["tw"], min_cutoff=1.0, beta=0.02)
        slot["f_th"] = OneEuroFilter(geom["th"], min_cutoff=1.0, beta=0.02)
        slot["f_ang"] = OneEuroFilter(geom["angle"], min_cutoff=1.5, beta=0.03)
        slot["f_wf"] = OneEuroFilter(geom["wf"], min_cutoff=1.0, beta=1.5)

    def _update_gain(self, frame_rgb, cx, cy, tw, th, t):
        """Sample face-region brightness for the skin/brightness match."""
        frame_h, frame_w = frame_rgb.shape[:2]
        x0 = max(int(cx - tw / 2), 0)
        y0 = max(int(cy - th / 2), 0)
        x1 = min(int(cx + tw / 2), frame_w)
        y1 = min(int(cy + th / 2), frame_h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        region = frame_rgb[y0:y1, x0:x1]
        scale = min(1.0, 48.0 / max(region.shape[0], region.shape[1]))
        if scale < 1.0:
            region = cv2.resize(region, (max(2, int(region.shape[1] * scale)),
                                         max(2, int(region.shape[0] * scale))))
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        face_luma = float((region.astype(np.float32) @ weights).mean())
        target = min(max(face_luma / max(self._overlay_luma, 1.0), 0.72), 1.45)
        if self._f_gain is None:
            self._f_gain = OneEuroFilter(1.0, min_cutoff=0.6, beta=0.2)
        self._gain = self._f_gain.filter(target, t)

    def _composite_all(self, frame_rgb, faces, t, det_scale):
        with self._overlay_lock:
            present = self._overlay_present
        if not present or self._overlay_rgb is None:
            return
        for i, (kind, data) in enumerate(faces):
            slot = self._slots[i] if i < MAX_FACES else self._slots[0]
            if kind == "mesh":
                geom = self._face_geometry(data, frame_rgb.shape[1],
                                           frame_rgb.shape[0])
            else:
                x, y, w, h = data
                geom = {"cx": x + w / 2.0, "cy": y + h / 2.0,
                        "tw": w * self._scale * HAAR_PAD,
                        "th": h * self._scale * HAAR_PAD,
                        "angle": 0.0, "wf": 1.0}
            if geom is None:
                continue
            self._blend_one(frame_rgb, geom, slot, t, i)

    def _blend_one(self, frame_rgb: np.ndarray, geom, slot, t: float,
                   face_index: int) -> None:
        with self._overlay_lock:
            overlay_rgb = self._overlay_rgb
            overlay_alpha = self._overlay_alpha
            present = self._overlay_present
        if not present or overlay_rgb is None:
            return

        frame_h, frame_w = frame_rgb.shape[:2]

        if slot["f_cx"] is None:
            self._init_filters(geom, slot)

        # One-euro smoothing: rock steady when the head is still, no
        # perceptible lag during fast motion (adaptive cutoff).
        cx = slot["f_cx"].filter(geom["cx"], t)
        cy = slot["f_cy"].filter(geom["cy"], t)
        tw = slot["f_tw"].filter(geom["tw"], t) * slot["f_wf"].filter(geom["wf"], t)
        th = slot["f_th"].filter(geom["th"], t)
        angle = min(max(slot["f_ang"].filter(geom["angle"], t), -60.0), 60.0)

        target_w = max(2, int(round(tw)))
        target_h = max(2, int(round(th)))

        # Warp cache: resize/rotate/feather only rerun when the quantized
        # geometry actually changes - a still head costs almost nothing
        # because the filters make the parameters settle on fixed values.
        cache_key = (target_w, target_h, int(angle * 2.0))
        if cache_key != slot["cache_key"]:
            interp = (cv2.INTER_AREA
                      if target_w < overlay_rgb.shape[1] else cv2.INTER_LINEAR)
            ov = cv2.resize(overlay_rgb, (target_w, target_h),
                            interpolation=interp)
            al = cv2.resize(overlay_alpha, (target_w, target_h),
                            interpolation=interp)
            # Positive atan2 angle (right eye lower on screen) means the
            # overlay must rotate clockwise on screen; getRotationMatrix2D is
            # positive-counter-clockwise, hence the negated angle (verified
            # empirically). Pad BEFORE rotating so the rotated rect always
            # has room, and pad the two canvases DIFFERENTLY: the color pad
            # replicates the edge colors (so the feather ramp melts into
            # continuation colors instead of a black fringe), the alpha pad
            # stays zero (so the ramp reaches 0 at the canvas edge).
            rad = math.radians(abs(angle))
            sin_a, cos_a = math.sin(rad), math.cos(rad)
            k = int(max(3, round(min(target_w, target_h) * FEATHER_FRACTION)))
            if k % 2 == 0:
                k += 1
            ov = cv2.copyMakeBorder(ov, k, k, k, k, cv2.BORDER_REPLICATE)
            al = cv2.copyMakeBorder(al, k, k, k, k,
                                    cv2.BORDER_CONSTANT, value=0.0)
            # Feathered edges: a hard alpha cutout screams "pasted PNG"; a
            # soft falloff melts the overlay into the skin. The blur happens
            # inside the pad, so the ramp always reaches 0.
            al = cv2.GaussianBlur(al, (k, k), 0)
            rot_w = int(math.ceil(target_w * cos_a + target_h * sin_a)) + 2 * k
            rot_h = int(math.ceil(target_w * sin_a + target_h * cos_a)) + 2 * k
            rot_matrix = cv2.getRotationMatrix2D(
                (target_w / 2.0, target_h / 2.0), -angle, 1.0)
            rot_matrix[0, 2] += (rot_w - target_w) / 2.0
            rot_matrix[1, 2] += (rot_h - target_h) / 2.0
            ov = cv2.warpAffine(ov, rot_matrix, (rot_w, rot_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            al = cv2.warpAffine(al, rot_matrix, (rot_w, rot_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            slot["cache_key"] = cache_key
            slot["cache_ov"] = ov
            slot["cache_al"] = al

        ov = slot["cache_ov"]
        al = slot["cache_al"]

        if self._skin_match and face_index == 0 and self._frame_idx % 4 == 0:
            self._update_gain(frame_rgb, cx, cy, target_w, target_h, t)
        if self._skin_match and abs(self._gain - 1.0) > 0.02:
            ov = np.clip(ov.astype(np.float32) * self._gain,
                         0.0, 255.0).astype(np.uint8)

        # The cached canvas is rotation-centered on the same face-box center,
        # so placement stays a simple centered blit, clamped to the frame;
        # the overlay is cropped wherever it would fall outside.
        rot_w = ov.shape[1]
        rot_h = ov.shape[0]
        ox = int(round(cx - rot_w / 2.0))
        oy = int(round(cy - rot_h / 2.0))
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + rot_w, frame_w), min(oy + rot_h, frame_h)
        if x1 <= x0 or y1 <= y0:
            return

        ov = ov[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        alpha = al[y0 - oy:y1 - oy, x0 - ox:x1 - ox] * self._opacity

        roi = frame_rgb[y0:y1, x0:x1]
        a = alpha[..., np.newaxis]
        # Single rounded float blend: splitting into floor-truncated steps
        # can darken the faint-alpha fringe by 1-2 LSBs (a dark hairline on
        # light backgrounds).
        blended = np.rint(roi * (1.0 - a) + ov * a)
        frame_rgb[y0:y1, x0:x1] = blended.astype(np.uint8)

# --------------------------------------------------------------------- UI ---


def detect_face_box(rgb):
    """Single-face bounding box (pixels) in a still image.

    MediaPipe first, OpenCV Haar fallback. Returns (x, y, w, h) or None.
    Used to analyze uploaded overlay photos.
    """
    h, w = rgb.shape[:2]
    if _mp is not None:
        fm = None
        try:
            if hasattr(_mp, "solutions"):
                fm = _mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    min_detection_confidence=0.4)
            else:
                fm = _TasksMeshAdapter(static_mode=True)
            res = fm.process(rgb)
            if res is not None and res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                xs = [lm.x * w for lm in lms]
                ys = [lm.y * h for lm in lms]
                return (int(min(xs)), int(min(ys)),
                        int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        except Exception:
            pass
        finally:
            if fm is not None:
                try:
                    fm.close()
                except Exception:
                    pass
    if hasattr(cv2, "CascadeClassifier"):
        cascade_path = os.path.join(str(cv2.data.haarcascades),
                                    "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(cascade_path)
            boxes = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if boxes is not None and len(boxes):
                x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
                return int(x), int(y), int(bw), int(bh)
    return None


def auto_cutout(rgb, alpha):
    """Turn a plain photo into a face overlay: feathered person cutout.

    Detects the face in the uploaded image, then refines a person mask
    with GrabCut (face ellipse as foreground seed). If no face is found
    the image is returned unchanged.
    """
    box = detect_face_box(rgb)
    if box is None:
        return rgb, alpha
    h, w = rgb.shape[:2]
    x, y, fw, fh = box
    # Expand so head and shoulders stay inside the mask.
    ex = int(max(24, fw * 0.45))
    ey_top = int(max(24, fh * 0.55))
    ey_bot = int(max(24, fh * 0.85))
    x0, y0 = max(0, x - ex), max(0, y - ey_top)
    x1, y1 = min(w, x + fw + ex), min(h, y + fh + ey_bot)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return rgb, alpha

    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = min(1.0, 384.0 / max(ch, cw))
    if scale < 1.0:
        crop_s = cv2.resize(crop, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    else:
        crop_s = crop
    sh, sw = crop_s.shape[:2]

    mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    cv2.rectangle(mask, (int(2 * scale), int(2 * scale)),
                  (int(sw - 2 * scale), int(sh - 2 * scale)),
                  cv2.GC_PR_FGD, -1)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    fg = None
    try:
        cv2.grabCut(crop_s, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        pass
    # Degenerate cases (person fills the crop so background and foreground
    # color models collapse, or GrabCut unavailable): fall back to a
    # head-and-shoulders ellipse so the overlay never comes out empty.
    if fg is None or fg.mean() < 0.02:
        fg = np.zeros((sh, sw), np.uint8)
        cv2.ellipse(fg, (int(sw / 2), int(sh * 0.45)),
                    (int(sw * 0.42), int(sh * 0.5)), 0, 0, 360, 255, -1)
        fg = fg > 127

    fg_u8 = (fg * 255).astype(np.uint8)
    k = int(max(3, round(min(sh, sw) * 0.05)))
    if k % 2 == 0:
        k += 1
    fg_u8 = cv2.GaussianBlur(fg_u8, (k, k), 0)
    fg_full = cv2.resize(fg_u8, (cw, ch), interpolation=cv2.INTER_LINEAR)
    # Everything outside the person becomes transparent: the overlay is the
    # cutout, not the original photo with a hole in it.
    new_alpha = np.zeros_like(alpha)
    new_alpha[y0:y1, x0:x1] = fg_full.astype(np.float32) / 255.0
    return rgb, new_alpha


def probe_virtual_camera():
    """Open and immediately close a virtual camera to detect a usable driver.

    Works with the OBS Virtual Camera and Unity Capture backends; returns
    (ok, error_detail). Camera() with no backend tries every registered
    backend in order, so any installed driver satisfies the probe.
    """
    if pyvirtualcam is None:
        return False, "pyvirtualcam is not installed"
    try:
        cam = pyvirtualcam.Camera(640, 480, fps=30,
                                  fmt=pyvirtualcam.PixelFormat.RGB)
        cam.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


class MainWindow(QWidget):
    PREVIEW_W, PREVIEW_H = 780, 440

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceSpoof")
        self.setFixedSize(820, 650)
        self.setStyleSheet(STYLESHEET)

        self.thread = None
        self.scanner = None
        self._pending_overlay = None  # (rgb, alpha) applied when capture starts
        self._ready_cameras = []
        self._scanning = False
        self._no_signal_pixmap = self._make_no_signal_pixmap()
        self._vcam_ok, self._vcam_detail = probe_virtual_camera()
        self._last_frame = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(8)

        # ---- preview pane ----
        self.preview = QLabel()
        self.preview.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(
            "QLabel { border: 1px solid #1A1A1A; background-color: #050505; }")
        self.preview.setPixmap(self._no_signal_pixmap)
        root.addWidget(self.preview)

        # ---- controls row 1 ----
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.camera_box = QComboBox()
        self.camera_box.setFixedWidth(220)
        row1.addWidget(self.camera_box)

        self.resolution_box = QComboBox()
        self.resolution_box.setFixedWidth(120)
        self.resolution_box.addItem("720p", (1280, 720))
        self.resolution_box.addItem("1080p", (1920, 1080))
        self.resolution_box.setCurrentIndex(0)
        row1.addWidget(self.resolution_box)

        self.rescan_button = QPushButton("Rescan")
        self.rescan_button.setFixedSize(80, 26)
        self.rescan_button.setToolTip(
            "Re-detect cameras. Only devices that actually deliver frames are listed.")
        self.rescan_button.clicked.connect(self._start_scan)
        row1.addWidget(self.rescan_button)

        row1.addStretch()
        root.addLayout(row1)

        # ---- controls row 2 ----
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.overlay_button = QPushButton("Overlay")
        self.overlay_button.setFixedSize(140, 32)
        self.overlay_button.clicked.connect(self._on_pick_overlay)
        row2.addWidget(self.overlay_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setFixedSize(80, 32)
        self.clear_button.clicked.connect(self._on_clear_overlay)
        row2.addWidget(self.clear_button)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(85)
        self.opacity_slider.setFixedWidth(140)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        row2.addWidget(self.opacity_slider)

        self.opacity_label = QLabel("85%")
        row2.addWidget(self.opacity_label)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(50, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.setFixedWidth(140)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        row2.addWidget(self.scale_slider)

        self.scale_label = QLabel("100%")
        row2.addWidget(self.scale_label)

        self.skin_check = QCheckBox("Skin")
        self.skin_check.setToolTip(
            "Match the overlay's brightness to the lighting on your face.\n"
            "For photographic overlays; leave off for graphics/emoji masks.")
        self.skin_check.toggled.connect(self._on_skin_toggled)
        row2.addWidget(self.skin_check)

        row2.addStretch()
        root.addLayout(row2)

        # ---- controls row 3 ----
        row3 = QHBoxLayout()
        row3.setSpacing(8)

        self.start_button = QPushButton("START")
        self.start_button.setObjectName("startButton")
        self.start_button.setFixedSize(120, 36)
        self.start_button.clicked.connect(self._on_start)
        row3.addWidget(self.start_button)

        self.stop_button = QPushButton("STOP")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setFixedSize(120, 36)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop)
        row3.addWidget(self.stop_button)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet(f"color: {STATUS_COLORS['idle']};")
        row3.addWidget(self.status_label)

        row3.addStretch()
        root.addLayout(row3)

        # ---- controls row 4 ----
        row4 = QHBoxLayout()
        row4.setSpacing(8)

        self.autocut_check = QCheckBox("Auto-cutout")
        self.autocut_check.setChecked(True)
        self.autocut_check.setToolTip(
            "When loading a plain photo (no transparency), detect the face\n"
            "in it and cut the person out automatically.")
        row4.addWidget(self.autocut_check)

        self.hud_check = QCheckBox("Tracking HUD")
        self.hud_check.setToolTip(
            "Draw green detection boxes into the feed for debugging.\n"
            "The boxes are visible in the output - turn off for real use.")
        row4.addWidget(self.hud_check)

        self.snapshot_button = QPushButton("Snapshot")
        self.snapshot_button.setFixedSize(100, 26)
        self.snapshot_button.setToolTip(
            "Save the current composited frame as a PNG.")
        self.snapshot_button.clicked.connect(self._on_snapshot)
        row4.addWidget(self.snapshot_button)

        row4.addStretch()
        root.addLayout(row4)

        # ---- startup sequence ----
        self._check_virtual_camera()
        self._update_start_enabled()
        self._start_scan()

    # ---- virtual camera ----

    def _check_virtual_camera(self):
        if not self._vcam_ok:
            tip = (f"A virtual camera driver is required (OBS Virtual Camera "
                   f"or Unity Capture).\n\n{self._vcam_detail}\n\n"
                   f"OBS: {OBS_DOWNLOAD_URL}\nUnity Capture: {UNITY_CAPTURE_URL}")
            self.start_button.setToolTip(tip)
            self._set_status("Error: No virtual camera driver", "error")
        else:
            self.start_button.setToolTip(
                "Outputs to the first available virtual camera "
                "(OBS Virtual Camera or Unity Video Capture).")

    # ---- camera scanning ----

    def _start_scan(self):
        if self._scanning or self.thread is not None:
            return
        self._scanning = True
        self._ready_cameras = []
        self.camera_box.clear()
        self.camera_box.addItem("Scanning...", -1)
        self.camera_box.setEnabled(False)
        self._update_start_enabled()
        if self._vcam_ok:
            self._set_status("Scanning cameras...", "idle")
        old = self.scanner
        self.scanner = CameraScanner()
        self.scanner.scanned.connect(self._on_scan_finished)
        self.scanner.start()
        if old is not None:
            try:
                old.deleteLater()
            except RuntimeError:
                pass  # already destroyed

    def _on_scan_finished(self, cameras):
        self._scanning = False
        self._ready_cameras = cameras
        self.camera_box.clear()
        for cam in cameras:
            label = (f"Camera {cam['index']} - {cam['name']}"
                     if cam["name"] else f"Camera {cam['index']}")
            self.camera_box.addItem(label, cam["index"])
            if cam["name"]:
                self.camera_box.setItemData(
                    self.camera_box.count() - 1, cam["name"],
                    Qt.ItemDataRole.ToolTipRole)
        self.camera_box.setEnabled(True)
        if cameras:
            self.camera_box.setCurrentIndex(0)
            if self._vcam_ok:
                self._set_status("Idle", "idle")
        elif self._vcam_ok:
            self._set_status("Error: No camera found", "error")
            self.status_label.setToolTip(
                "No camera delivered frames. Close other apps that may be\n"
                "using the camera, check the privacy settings\n"
                "(Settings > Privacy > Camera), then click Rescan.")
        self._update_start_enabled()

    def _update_start_enabled(self):
        can_start = (not self._scanning and self.thread is None
                     and self._vcam_ok and bool(self._ready_cameras))
        self.start_button.setEnabled(can_start)
        self.rescan_button.setEnabled(not self._scanning and self.thread is None)
        self.camera_box.setEnabled(not self._scanning and self.thread is None)
        self.resolution_box.setEnabled(self.thread is None)
        self.stop_button.setEnabled(self.thread is not None)

    @staticmethod
    def _make_no_signal_pixmap():
        pix = QPixmap(780, 440)
        pix.fill(QColor("#050505"))
        painter = QPainter(pix)
        painter.setPen(QColor("#333333"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "NO SIGNAL")
        painter.end()
        return pix

    # ---- status ----

    def _set_status(self, text: str, kind: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {STATUS_COLORS[kind]};")

    # ---- overlay handling ----

    def _on_pick_overlay(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select overlay image", "",
            "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.overlay_button.setToolTip(path)
        try:
            image = Image.open(path).convert("RGBA")
            arr = np.asarray(image)
            rgb = np.ascontiguousarray(arr[:, :, :3])
            alpha = arr[:, :, 3].astype(np.float32) / 255.0
        except Exception:
            self._pending_overlay = None
            if self.thread is not None:
                self.thread.clear_overlay()
            self._set_status("Error: Invalid overlay", "error")
            return

        if self.autocut_check.isChecked() and float(alpha.mean()) > 0.97:
            self._set_status("Analyzing overlay face...", "idle")
            QApplication.processEvents()
            try:
                rgb, alpha = auto_cutout(rgb, alpha)
            except Exception:
                pass  # keep the raw image if analysis fails

        if self.thread is not None:
            self.thread.set_overlay(rgb, alpha)
        else:
            self._pending_overlay = (rgb, alpha)

    def _on_clear_overlay(self):
        self._pending_overlay = None
        self.overlay_button.setToolTip("")
        if self.thread is not None:
            self.thread.clear_overlay()

    def _on_snapshot(self):
        if self._last_frame is None:
            self._set_status("Error: Nothing to snapshot yet", "error")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", "facespoof_snapshot.png", "PNG (*.png)")
        if not path:
            return
        try:
            Image.fromarray(self._last_frame).save(path)
            self._set_status(f"Saved {os.path.basename(path)}", "idle")
        except Exception:
            self._set_status("Error: Snapshot failed", "error")

    def _on_opacity_changed(self, value: int):
        self.opacity_label.setText(f"{value}%")
        if self.thread is not None:
            self.thread.set_opacity(value)

    def _on_scale_changed(self, value: int):
        self.scale_label.setText(f"{value}%")
        if self.thread is not None:
            self.thread.set_scale(value)

    def _on_skin_toggled(self, checked: bool):
        if self.thread is not None:
            self.thread.set_skin_match(checked)

    # ---- start / stop ----

    def _on_start(self):
        if self._scanning or self.thread is not None or not self._vcam_ok:
            return
        index = self.camera_box.currentData()
        if index is None or index < 0:
            return
        width, height = self.resolution_box.currentData()

        self.thread = CaptureThread(index, width, height)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.status_changed.connect(self._on_worker_status)
        self.thread.finished.connect(self._on_thread_finished)

        self._update_start_enabled()

        self.thread.set_opacity(self.opacity_slider.value())
        self.thread.set_scale(self.scale_slider.value())
        self.thread.set_skin_match(self.skin_check.isChecked())
        self.thread.set_show_tracking(self.hud_check.isChecked())
        if self._pending_overlay is not None:
            rgb, alpha = self._pending_overlay
            self.thread.set_overlay(rgb, alpha)

        self.thread.start()

    def _on_stop(self):
        if self.thread is not None:
            self.thread.stop()

    def _on_worker_status(self, text: str, kind: str):
        self._set_status(text, "error" if kind.startswith("error") else kind)
        if text == "Error: No virtual camera driver":
            self.status_label.setToolTip(
                f"Install OBS: {OBS_DOWNLOAD_URL}\n"
                f"Or Unity Capture: {UNITY_CAPTURE_URL}")
        if kind == "error" and self.thread is not None:
            # Fatal errors end the loop; the thread emits finished right
            # after and button state is restored in _on_thread_finished.
            # "error-soft" (face mesh failed) keeps streaming in passthrough.
            self.thread.stop()

    def _on_thread_finished(self):
        self.thread = None
        self._update_start_enabled()
        if self.status_label.text() == "Running":
            self._set_status("Idle", "idle")
        self.preview.setPixmap(self._no_signal_pixmap)

    # ---- preview rendering ----

    def _on_frame(self, frame: np.ndarray):
        self._last_frame = frame
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, 3 * width,
                       QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.PREVIEW_W, self.PREVIEW_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)
        self.preview.setPixmap(scaled)

    # ---- shutdown ----

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(3000)
        if self.scanner is not None:
            try:
                self.scanner.abort()
                self.scanner.wait(3000)
            except RuntimeError:
                pass  # already destroyed
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FaceSpoof")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
