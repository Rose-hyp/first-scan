"""FaceSpoof — real-time face-tracked PNG overlay rendered to a virtual camera.

Pipeline:
    physical camera -> OpenCV capture -> MediaPipe Face Mesh (468 landmarks)
    -> alpha-blended PNG overlay positioned/rotated/scaled to the face
    -> pyvirtualcam sink (OBS Virtual Camera or Unity Capture)

Windows only. Requires Python 3.10-3.12 64-bit.
"""

import math
import sys
import threading

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication,
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
"""

STATUS_COLORS = {
    "idle": "#666666",
    "running": "#7ACC7A",
    "error": "#CC7A7A",
}


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

        self._opacity = 0.85
        self._scale = 1.0
        self._passthrough = False
        self._face_mesh = None

    # ---- controls called from the UI thread ----

    def set_overlay(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        with self._overlay_lock:
            self._overlay_rgb = rgb
            self._overlay_alpha = alpha
            self._overlay_present = True

    def clear_overlay(self) -> None:
        with self._overlay_lock:
            self._overlay_rgb = None
            self._overlay_alpha = None
            self._overlay_present = False

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
                self._face_mesh = _mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
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

            # Drivers sometimes ignore the requested resolution; normalize so
            # the virtual camera dimensions always match the pushed frames.
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height),
                                   interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not self._passthrough and self._overlay_present:
                result = self._face_mesh.process(rgb)
                landmarks = (result.multi_face_landmarks[0].landmark
                             if result.multi_face_landmarks else None)
                if landmarks is not None:
                    self._composite(rgb, landmarks)
                # No face: emit the untouched frame.

            self.frame_ready.emit(rgb)
            vcam.send(rgb)
            vcam.sleep_until_next_frame()

        cap.release()
        vcam.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    # ---- compositing ----

    def _composite(self, frame_rgb: np.ndarray, landmarks) -> None:
        with self._overlay_lock:
            overlay_rgb = self._overlay_rgb
            overlay_alpha = self._overlay_alpha
            present = self._overlay_present
        if not present or overlay_rgb is None:
            return

        frame_h, frame_w = frame_rgb.shape[:2]

        # Face bounding box from all 468 landmarks (normalized -> pixels).
        xs = np.fromiter((lm.x for lm in landmarks), dtype=np.float64)
        ys = np.fromiter((lm.y for lm in landmarks), dtype=np.float64)
        x_min, x_max = xs.min() * frame_w, xs.max() * frame_w
        y_min, y_max = ys.min() * frame_h, ys.max() * frame_h
        face_w = x_max - x_min
        face_h = y_max - y_min
        if face_w < 4 or face_h < 4:
            return

        # Rotation from the eye line: left eye = 33/133, right eye = 362/263.
        lx = (landmarks[33].x + landmarks[133].x) * 0.5 * frame_w
        ly = (landmarks[33].y + landmarks[133].y) * 0.5 * frame_h
        rx = (landmarks[362].x + landmarks[263].x) * 0.5 * frame_w
        ry = (landmarks[362].y + landmarks[263].y) * 0.5 * frame_h
        angle = math.degrees(math.atan2(ry - ly, rx - lx))

        # Target size: face box * user scale * 1.2 padding factor.
        target_w = max(2, int(round(face_w * self._scale * 1.2)))
        target_h = max(2, int(round(face_h * self._scale * 1.2)))

        interp = cv2.INTER_AREA if target_w < overlay_rgb.shape[1] else cv2.INTER_LINEAR
        ov = cv2.resize(overlay_rgb, (target_w, target_h), interpolation=interp)
        al = cv2.resize(overlay_alpha, (target_w, target_h), interpolation=interp)

        # Positive atan2 angle (right eye lower on screen) means the overlay
        # must rotate clockwise on screen; getRotationMatrix2D is positive-
        # counter-clockwise, hence the negated angle (verified empirically).
        rot_matrix = cv2.getRotationMatrix2D(
            (target_w / 2.0, target_h / 2.0), -angle, 1.0)
        ov = cv2.warpAffine(ov, rot_matrix, (target_w, target_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        al = cv2.warpAffine(al, rot_matrix, (target_w, target_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Position centered on the face box, clamped to the frame; the overlay
        # is cropped wherever it would fall outside.
        ox = int(round((x_min + x_max) / 2.0 - target_w / 2.0))
        oy = int(round((y_min + y_max) / 2.0 - target_h / 2.0))
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + target_w, frame_w), min(oy + target_h, frame_h)
        if x1 <= x0 or y1 <= y0:
            return

        ov = ov[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        alpha = al[y0 - oy:y1 - oy, x0 - ox:x1 - ox] * self._opacity

        roi = frame_rgb[y0:y1, x0:x1]
        a = alpha[..., np.newaxis]
        np.multiply(roi, 1.0 - a, out=roi, casting="unsafe")
        roi += (ov * a).astype(np.uint8)


# --------------------------------------------------------------------- UI ---


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
        self.setFixedSize(820, 620)
        self.setStyleSheet(STYLESHEET)

        self.thread = None
        self.scanner = None
        self._pending_overlay = None  # (rgb, alpha) applied when capture starts
        self._ready_cameras = []
        self._scanning = False
        self._no_signal_pixmap = self._make_no_signal_pixmap()
        self._vcam_ok, self._vcam_detail = probe_virtual_camera()

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

        if self.thread is not None:
            self.thread.set_overlay(rgb, alpha)
        else:
            self._pending_overlay = (rgb, alpha)

    def _on_clear_overlay(self):
        self._pending_overlay = None
        self.overlay_button.setToolTip("")
        if self.thread is not None:
            self.thread.clear_overlay()

    def _on_opacity_changed(self, value: int):
        self.opacity_label.setText(f"{value}%")
        if self.thread is not None:
            self.thread.set_opacity(value)

    def _on_scale_changed(self, value: int):
        self.scale_label.setText(f"{value}%")
        if self.thread is not None:
            self.thread.set_scale(value)

    # ---- start / stop ----

    def _on_start(self):
        if self._scanning or self.thread is not None:
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
