@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [!] No .venv found. Run setup.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe facespoof.py
if errorlevel 1 pause
