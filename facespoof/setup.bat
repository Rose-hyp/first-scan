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

echo.
echo [+] Setup complete.
echo     Run the app:     run.bat
echo     Build the .exe:  build.bat
pause
