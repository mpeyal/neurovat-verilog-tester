@echo off
REM virtuoso_connect.bat -- open the SSH tunnel to Virtuoso and run the bridge test.
REM Uses SSH key auth (no password). Prereq on the Linux side: Virtuoso is open
REM and the skillbridge server is running (skill() in the CIW).

setlocal
set HOST=mahmudulpeyal@coen-cassia.boisestate.edu
REM per-user socket (the shared -default.sock is squatted by another account on
REM this multi-user host); skill()/pyStartServer use ?id = login -> this socket.
set TUNNEL=7777:/tmp/skill-server-mahmudulpeyal.sock

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
