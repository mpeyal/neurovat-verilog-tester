@echo off
rem One-click setup and launch for Windows. Double-click this file: the first
rem run creates .venv and installs requirements.txt, later runs just start
rem NeuroVAT (they reinstall only when requirements.txt changes).
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"

if exist "%VPY%" goto install

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (where python >nul 2>&1 && set "PY=python")
if not defined PY goto nopython
%PY% -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1 || goto nopython
echo Creating virtual environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 goto fail

:install
fc /b requirements.txt .venv\requirements.installed >nul 2>&1 && goto run
echo Installing requirements, this takes a few minutes the first time ...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail
copy /y requirements.txt .venv\requirements.installed >nul

:run
echo Starting NeuroVAT ...
"%VPY%" run_gui.py %*
if errorlevel 1 pause
exit /b

:nopython
echo Python 3.9 or newer is required. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then run this file again.
pause
exit /b 1

:fail
echo Setup failed; see the messages above.
pause
exit /b 1
