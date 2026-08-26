# FaceSpoof

Real-time, face-tracked PNG overlay (mask, avatar, filter) rendered on top of
your physical webcam feed and pushed to a virtual camera. The composited feed
shows up in any app that uses a camera: OBS, Discord, browsers, conferencing
software — the device appears under its real driver name (e.g. *OBS Virtual
Camera*).

Useful for streaming, VTubing, or keeping your actual face off recorded calls
by presenting an overlay instead.

## Requirements

- Windows 10/11
- Python 3.10, 3.11 or 3.12 — **64-bit** (MediaPipe has no wheels outside 3.9–3.12)
- A virtual camera driver — `setup.bat` **auto-installs
  [Unity Capture](https://github.com/schellingb/UnityCapture)** (a ~2 MB
  DirectShow filter, no OBS required). OBS Studio's "OBS Virtual Camera" is
  also supported if you prefer it.

The app tries every registered virtual camera backend automatically and uses
the first that works. Unity Capture appears to other apps as
**"Unity Video Capture"**; OBS's appears as "OBS Virtual Camera".
To uninstall Unity Capture later: `regsvr32 /u "%LocalAppData%\UnityCapture\UnityCaptureFilter64.dll"`

## Setup (easy way)

Double-click **`setup.bat`**. It finds a suitable Python (3.10–3.12, 64-bit),
creates `.venv` and installs everything. Then use:

- **`run.bat`** — start the app
- **`build.bat`** — compile `dist\FaceSpoof.exe`

If setup.bat finds no suitable Python it **downloads Python 3.12.10 64-bit
from python.org and installs it automatically** — per-user, silently, with
PATH configured. No admin rights, no manual steps, existing Python installs
are left alone. setup.bat also finds Pythons that are *not* on PATH (registry
or standard install folders) and adds them to your user PATH by itself.

## Setup (manual)

Run each of these as its **own command** — do not paste them as a single line:

```bat
py -3.12-64 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If `py -3.12-64` reports no runtime, try `py -3.11-64`, `py -3.10-64`, or see
what is registered with `py -0`. Then run the app from the same prompt:

```bat
python facespoof.py
```

## Run from source

```bat
python facespoof.py
```

On startup the app enumerates camera indices 0–9 (may take a few seconds) and
checks for a virtual camera driver. If no driver is found, START is disabled
until OBS (or Unity Capture) is installed.

## Build the .exe

```bat
build.bat
```

(or manually: `python -m pip install pyinstaller`, then
`python -m PyInstaller --onefile --noconsole --name FaceSpoof --add-data "mediapipe;mediapipe" --add-data "cv2;cv2" facespoof.py`)

Output: `dist\FaceSpoof.exe` — standalone, no Python installation needed on
the target machine.

## Controls

| Control | Function |
|---|---|
| Camera dropdown | **Active cameras only** — every listed device was probed and actually delivered frames. Shows real device names (e.g. `Camera 0 - HD Webcam`), newest scan via **Rescan** |
| Rescan | Re-detect cameras without restarting |
| Resolution | 720p (default) or 1080p |
| Overlay | Load a `.png`/`.jpg`/`.jpeg` — plain photos get an **auto-cutout** (the face is detected and the person is cut out automatically; untick *Auto-cutout* to skip) |
| Clear | Remove the active overlay |
| Opacity slider | 0–100 % blend of the overlay over the face (default 85) |
| Scale slider | 50–200 % of the auto-computed face-box size (default 100) |
| Tracking HUD | Draws green boxes around every detected face — debugging aid; **the boxes are visible in the output**, turn off for real use |
| Snapshot | Save the current composited frame as a PNG |
| START / STOP | Open or release camera + virtual camera sink |

## How the overlay engine matches the face

Alignment is built from stable landmark anchors, then temporally filtered:

1. **Tracking filters** — face center, size, rotation and yaw are smoothed
   with a **1-euro filter** (the same adaptive filter Google's FaceLandmarker
   uses in streaming mode): no visible jitter while still, no perceptible
   lag when you move fast (measured: 2.8x jitter reduction, 4-frame settle
   after a jump vs 16 frames for a fixed filter).
2. **Yaw behavior** — the overlay's center follows the nose tip and its width
   narrows as the head turns, so it stays glued to the face in profile.
3. **Feathered edges** — the alpha mask is gaussian-feathered (~6 % of
   overlay size) on a padded, rotation-expanded canvas so the melt into the
   background is smooth at every angle and edges never clip (verified free of
   dark fringes down to single-pixel rounding).
4. **Skin match (optional, `Skin` checkbox)** — matches the overlay's
   brightness to the lighting on your face; for photographic overlays.
5. **Speed** — Face Mesh runs on a 640 px-downscaled copy (landmarks are
   normalized, so coordinates are unchanged), and the resize/rotate/feather
   pipeline is cached against quantized geometry: a still head costs almost
   nothing. Measured compositing cost: ~5.7 ms/frame at 720p (30 fps budget:
   33 ms).

## How camera detection works

On startup and on **Rescan**, a background thread enumerates DirectShow video
input devices (via `pygrabber`, which is why real names appear), then probes
each one: a device is listed **only** if it can be opened *and* returns a real
frame. Cameras held or disabled elsewhere never show up as fake entries.
Capture itself opens with DirectShow and falls back to Media Foundation for
cameras that need it, requests MJPG for full-rate 1080p, and tolerates brief
read glitches instead of dying on the first dropped frame.

## How the tracking works

Detection is **tiered so it never silently fails**:

1. **MediaPipe face landmarks (468 points)** run per frame on a 640 px
   downscaled copy. Both mediapipe generations are supported: the classic
   `solutions` API, and — on mediapipe ≥ 0.10.35 which removed it — the
   newer **Tasks FaceLandmarker**, whose model FaceSpoof downloads once to
   `%LocalAppData%\facespoof` automatically.
2. **OpenCV Haar-cascade fallback** — if MediaPipe is unavailable, crashes,
   or finds nothing (small/off-angle faces), an OpenCV cascade detector
   takes over the same frame. This tier keeps working even if MediaPipe
   fails to initialize completely.
3. **Adaptive retry** — while a face is considered lost, detection
   periodically retries at full resolution before giving up.

Up to **3 faces** are tracked simultaneously — each gets its own filter and
warp-cache slot, so several people can each wear the overlay. Per face the
overlay is:

1. sized to the detected face bounding box × scale × 1.2 padding,
2. rotated to the eye line (landmarks 33/133 vs 362/263) via `warpAffine`,
3. centered on the face box, cropped at frame edges,
4. alpha-blended at the chosen opacity.

If no face is detected the raw frame passes through untouched (enable the
**Tracking HUD** to see exactly what the detector sees).

## Troubleshooting

- **START disabled, "Error: No virtual camera driver"** — the *output* side is
  missing, not your camera (the source dropdown lists your physical cameras).
  Re-run `setup.bat` — it installs and registers the Unity Capture filter
  automatically (one UAC prompt). Hover the START button or the status label
  to see the exact driver error.
- **"Error: No device at index N"** — the camera is held by another
  application or the index vanished; pick another entry.
- **Black preview but virtual camera works** — some drivers ignore requested
  resolutions; frames are resized internally so the sink stays consistent.
