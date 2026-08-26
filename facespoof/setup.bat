@echo off
setlocal
cd /d "%~dp0"
title FaceSpoof setup

rem -------------------------------------------------------------------
rem  FaceSpoof setup - zero manual steps:
rem    1. finds a Python 3.10-3.12 64-bit anywhere on the machine
rem       (py launcher, PATH, registry, standard install folders)
rem    2. adds it to your user PATH automatically (persistent)
rem    3. if none exists, downloads Python 3.12.10 64-bit from python.org
rem       and installs it per-user, silently, with PATH preconfigured
rem    4. creates .venv and installs all dependencies
rem -------------------------------------------------------------------

set "PROBE=import sys,struct;sys.exit(0 if struct.calcsize('P')==8 and (3,9)<=sys.version_info[:2]<=(3,12) else 1)"
set "PYEXE="

echo [*] Searching for a Python 3.10-3.12 64-bit installation ...

rem ---- 1) py launcher ----
for %%V in (3.12-64 3.11-64 3.10-64 3.12 3.11 3.10) do if not defined PYEXE (
    py -%%V -c "%PROBE%" >nul 2>nul && for /f "tokens=*" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
)

rem ---- 2) python on PATH ----
if not defined PYEXE (
    python -c "%PROBE%" >nul 2>nul && for /f "tokens=*" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
)

rem ---- 3) registry - PEP 514 entries written by every python.org installer ----
for %%H in (HKLM HKCU) do for %%V in (3.12 3.11 3.10) do if not defined PYEXE (
    for /f "tokens=2*" %%A in ('reg query "%%H\Software\Python\PythonCore\%%V\InstallPath" /ve 2^>nul') do (
        if /i "%%A"=="REG_SZ" if exist "%%B\python.exe" (
            "%%B\python.exe" -c "%PROBE%" >nul 2>nul && set "PYEXE=%%B\python.exe"
        )
    )
)

rem ---- 4) standard install folders, even with no PATH and no registry ----
for %%R in ("%LocalAppData%\Programs\Python" "C:\Program Files\Python" "C:\Program Files") do (
    for /d %%D in ("%%~R\Python3*") do if exist "%%D\python.exe" (
        "%%D\python.exe" -c "%PROBE%" >nul 2>nul && if not defined PYEXE set "PYEXE=%%D\python.exe"
    )
)

rem ---- 5) nothing suitable: silently install Python 3.12.10 64-bit ----
if defined PYEXE goto python_ready

echo [!] No suitable Python found - your existing Python installs stay untouched.
echo [*] Downloading Python 3.12.10 64-bit from python.org ...
set "PYSRC=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PYSETUP=%TEMP%\python-3.12.10-amd64.exe"
curl.exe -fL -o "%PYSETUP%" "%PYSRC%" >nul 2>nul
if not errorlevel 1 goto have_installer
echo [*] curl not available - downloading via PowerShell ...
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYSRC%' -OutFile \"$env:TEMP\python-3.12.10-amd64.exe\""
if not exist "%PYSETUP%" (
    echo [!] Download failed. Check your internet connection and re-run setup.bat
    pause
    exit /b 1
)
:have_installer
echo [*] Installing silently - per-user, no admin, PATH enabled automatically ...
"%PYSETUP%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
set /a tries=0
:waitinstall
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" goto install_done
set /a tries+=1
if %tries% gtr 40 (
    echo [!] Installation did not finish in time. Run the installer manually:
    echo     %PYSETUP%
    pause
    exit /b 1
)
timeout /t 3 /nobreak >nul
goto waitinstall
:install_done
set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
echo [+] Python 3.12.10 installed and registered on your PATH.

:python_ready
rem ---- 6) persist the chosen python dir on the USER PATH if missing ----
for %%P in ("%PYEXE%") do set "PYDIR=%%~dpP"
if "%PYDIR:~-1%"=="\" set "PYDIR=%PYDIR:~0,-1%"
echo %PATH%| find /i "%PYDIR%" >nul
if errorlevel 1 goto fix_path
echo [+] Already on PATH: %PYDIR%
goto path_done
:fix_path
powershell -NoProfile -Command "$p=[Environment]::GetEnvironmentVariable('Path','User'); if(-not $p){$p=''}; if(($p -split ';') -notcontains '%PYDIR%'){[Environment]::SetEnvironmentVariable('Path', ($p.TrimEnd(';') + ';%PYDIR%'), 'User'); Write-Host ('[+] Added to user PATH: ' + '%PYDIR%')}"
set "PATH=%PATH%;%PYDIR%"
echo [+] Usable in this window and in every new terminal.
:path_done

echo [*] Using Python: %PYEXE%
"%PYEXE%" -c "import sys;print('    ' + sys.version)"

rem ---- 7) fresh .venv + dependencies ----
if exist ".venv\Scripts\python.exe" (
    echo [*] Removing old .venv ...
    rmdir /s /q .venv
)
echo [*] Creating virtual environment ...
"%PYEXE%" -m venv .venv
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

rem ---- 8) virtual camera driver - required for the app to output anything ----
echo [*] Checking virtual camera driver ...
.venv\Scripts\python.exe -c "import pyvirtualcam,sys;sys.exit(0 if pyvirtualcam.camera_count()>0 else 1)" >nul 2>nul
if not errorlevel 1 (
    echo [+] Virtual camera driver found.
    goto setup_done
)
echo [!] No virtual camera driver. The app needs OBS Virtual Camera as its output.
set "ANSWER="
set /p ANSWER=Download and install OBS Studio automatically, about 150 MB [Y/n]: 
if /i "%ANSWER%"=="n" goto skip_obs
echo [*] Finding the latest OBS Studio release ...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $r=Invoke-RestMethod 'https://api.github.com/repos/obsproject/obs-studio/releases/latest'; $u=($r.assets | Where-Object { $_.name -like '*Full-Installer-x64.exe' } | Select-Object -First 1).browser_download_url; Write-Host ('    ' + $u); Invoke-WebRequest -Uri $u -OutFile ($env:TEMP + '\obs-setup.exe')"
if not exist "%TEMP%\obs-setup.exe" (
    echo [!] Download failed. Install OBS manually: https://obsproject.com/
    goto setup_done
)
echo [*] Installing OBS Studio - approve the Windows UAC prompt ...
powershell -NoProfile -Command "Start-Process -FilePath ($env:TEMP + '\obs-setup.exe') -ArgumentList '/VERYSILENT','/NORESTART','/SUPPRESSMSGBOXES' -Verb RunAs -Wait"
.venv\Scripts\python.exe -c "import pyvirtualcam,sys;sys.exit(0 if pyvirtualcam.camera_count()>0 else 1)" >nul 2>nul
if errorlevel 1 (
    echo [!] OBS installed but the virtual camera is not registered yet.
    echo     Open OBS once, or reboot, then run run.bat again.
) else (
    echo [+] OBS Virtual Camera ready.
)
goto setup_done
:skip_obs
echo [!] Skipped. START stays disabled until a virtual camera driver is installed.
echo     Get OBS from: https://obsproject.com/
:setup_done
echo.
echo [+] Setup complete.
echo     Run the app:     run.bat
echo     Build the .exe:  build.bat
pause
