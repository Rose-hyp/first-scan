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
| Overlay | Load a `.png`/`.jpg`/`.jpeg` with alpha (JPEGs get full opacity) |
| Clear | Remove the active overlay |
| Opacity slider | 0–100 % blend of the overlay over the face (default 85) |
| Scale slider | 50–200 % of the auto-computed face-box size (default 100) |
| START / STOP | Open or release camera + virtual camera sink |

## How camera detection works

On startup and on **Rescan**, a background thread enumerates DirectShow video
input devices (via `pygrabber`, which is why real names appear), then probes
each one: a device is listed **only** if it can be opened *and* returns a real
frame. Cameras held or disabled elsewhere never show up as fake entries.
Capture itself opens with DirectShow and falls back to Media Foundation for
cameras that need it, requests MJPG for full-rate 1080p, and tolerates brief
read glitches instead of dying on the first dropped frame.

## How the tracking works

MediaPipe Face Mesh (468 landmarks) runs per frame. The overlay is:

1. sized to the detected face bounding box × scale × 1.2 padding,
2. rotated to the eye line (landmarks 33/133 vs 362/263) via `warpAffine`,
3. centered on the face box, cropped at frame edges,
4. alpha-blended at the chosen opacity.

If no face is detected the raw frame passes through untouched. If MediaPipe
fails to initialize the app keeps streaming in passthrough (no overlay).

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
