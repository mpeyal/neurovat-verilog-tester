@echo off
REM virtuoso_connect.bat -- open the SSH tunnel to Virtuoso and run the bridge test.
REM Uses SSH key auth (no password). Prereq on the Linux side: Virtuoso is open
REM and the skillbridge server is running (skill() in the CIW).

setlocal
REM The Virtuoso login (user@host) is kept OUT of the repo: set the
REM NVAT_VIRTUOSO_HOST env var, or put one "user@host" line in virtuoso.local.
if not defined NVAT_VIRTUOSO_HOST if exist "%~dp0virtuoso.local" set /p NVAT_VIRTUOSO_HOST=<"%~dp0virtuoso.local"
if not defined NVAT_VIRTUOSO_HOST (
    echo Set NVAT_VIRTUOSO_HOST=user@host  ^(or create virtuoso.local with that line^).
    pause
    exit /b 1
)
set HOST=%NVAT_VIRTUOSO_HOST%
REM per-user socket; skill()/pyStartServer use ?id = login -> this socket.
for /f "tokens=1 delims=@" %%u in ("%HOST%") do set VUSER=%%u
set TUNNEL=7777:/tmp/skill-server-%VUSER%.sock

REM If something is already listening on 7777, assume the tunnel is up.
powershell -NoProfile -Command "if (Test-NetConnection -ComputerName localhost -Port 7777 -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo Tunnel already running on port 7777 -- reusing it.
    goto :test
)

echo Starting SSH tunnel...
start "Virtuoso SSH tunnel (leave open)" /min ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L %TUNNEL% %HOST%

REM Wait up to ~15s for the tunnel to come up.
for /l %%i in (1,1,15) do (
    powershell -NoProfile -Command "if (Test-NetConnection -ComputerName localhost -Port 7777 -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto :test
    timeout /t 1 /nobreak >nul
)
echo [FAIL] Tunnel did not come up within 15 seconds.
pause
exit /b 1

:test
echo.
echo Running connection test...
python "%~dp0connect_test.py"
echo.
pause
