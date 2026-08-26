@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [!] No .venv found. Run setup.bat first.
    pause
    exit /b 1
)

echo [*] Ensuring PyInstaller ...
.venv\Scripts\python.exe -m pip show pyinstaller >nul 2>nul || .venv\Scripts\python.exe -m pip install pyinstaller

echo [*] Building ...
.venv\Scripts\python.exe -m PyInstaller --onefile --noconsole --name FaceSpoof ^
  --add-data "mediapipe;mediapipe" ^
  --add-data "cv2;cv2" ^
  facespoof.py

if errorlevel 1 (
    echo [!] Build failed. Check the error above.
    pause
    exit /b 1
)

echo.
echo [+] Done. Output: dist\FaceSpoof.exe
pause
