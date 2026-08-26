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
- Python 3.10, 64-bit
- A virtual camera driver — install [OBS Studio](https://obsproject.com/)
  (its bundled "OBS Virtual Camera" is the default sink)

## Setup

```bat
py -3.10-64 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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
pip install pyinstaller
build.bat
```

Output: `dist\FaceSpoof.exe` — standalone, no Python installation needed on
the target machine.

## Controls

| Control | Function |
|---|---|
| Camera dropdown | Physical capture device (indices 0–9) |
| Resolution | 720p (default) or 1080p |
| Overlay | Load a `.png`/`.jpg`/`.jpeg` with alpha (JPEGs get full opacity) |
| Clear | Remove the active overlay |
| Opacity slider | 0–100 % blend of the overlay over the face (default 85) |
| Scale slider | 50–200 % of the auto-computed face-box size (default 100) |
| START / STOP | Open or release camera + virtual camera sink |

## How the tracking works

MediaPipe Face Mesh (468 landmarks) runs per frame. The overlay is:

1. sized to the detected face bounding box × scale × 1.2 padding,
2. rotated to the eye line (landmarks 33/133 vs 362/263) via `warpAffine`,
3. centered on the face box, cropped at frame edges,
4. alpha-blended at the chosen opacity.

If no face is detected the raw frame passes through untouched. If MediaPipe
fails to initialize the app keeps streaming in passthrough (no overlay).

## Troubleshooting

- **START disabled, "Error: No virtual camera driver"** — install OBS Studio,
  then restart the app. https://obsproject.com/
- **"Error: No device at index N"** — the camera is held by another
  application or the index vanished; pick another entry.
- **Black preview but virtual camera works** — some drivers ignore requested
  resolutions; frames are resized internally so the sink stays consistent.
