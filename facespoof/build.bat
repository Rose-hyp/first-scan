@echo off
REM FaceSpoof build script - run from an activated Python 3.10 64-bit env.
REM Requires: pip install -r requirements.txt && pip install pyinstaller

pyinstaller --onefile --noconsole --name FaceSpoof ^
  --add-data "mediapipe;mediapipe" ^
  --add-data "cv2;cv2" ^
  facespoof.py

echo.
echo Output: dist\FaceSpoof.exe
