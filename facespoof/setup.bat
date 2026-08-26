@echo off
setlocal
cd /d "%~dp0"

rem ---------------------------------------------------------------
rem FaceSpoof setup: finds a suitable Python 3.10-3.12 64-bit,
rem creates a local .venv and installs all dependencies.
rem ---------------------------------------------------------------

set "PROBE=import sys,struct;sys.exit(0 if struct.calcsize('P')==8 and (3,9)<=sys.version_info[:2]<=(3,12) else 1)"

set "PYCMD="
for %%V in (3.12-64 3.11-64 3.10-64 3.12 3.11 3.10) do (
    if not defined PYCMD (
        py -%%V -c "%PROBE%" >nul 2>nul && set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "%PROBE%" >nul 2>nul && set "PYCMD=python"
)

if not defined PYCMD (
    echo [!] No suitable Python found.
    echo     Install Python 3.10, 3.11 or 3.12 - 64-bit - from:
    echo     https://www.python.org/downloads/
    echo     IMPORTANT: tick "Add python.exe to PATH" in the installer.
    echo     Then run this script again.
    pause
    exit /b 1
)

echo [*] Using Python: %PYCMD%
%PYCMD% -c "import sys;print('    ' + sys.version)"

if exist ".venv\Scripts\python.exe" (
    echo [*] Removing old .venv ...
    rmdir /s /q .venv
)

echo [*] Creating virtual environment ...
%PYCMD% -m venv .venv
if errorlevel 1 (
    echo [!] venv creation failed.
    pause
    exit /b 1
)

echo [*] Upgrading pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [*] Installing dependencies - this can take a few minutes ...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [!] Dependency installation failed. Check the error above.
    pause
    exit /b 1
)

echo.
echo [+] Setup complete.
echo     Run the app:     run.bat
echo     Build the .exe:  build.bat
pause
