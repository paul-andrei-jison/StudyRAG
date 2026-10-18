@echo off
rem Study Buddy launcher: checks LM Studio and the models, then starts the app.
rem   start.bat          check, and ask before starting if something is wrong
rem   start.bat wait     for "start with Windows": retry the check for up to 10 minutes while LM Studio loads
rem Any other words (for example --no-bot) are passed to the app.
setlocal
cd /d "%~dp0"
title Study Buddy

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Python for Study Buddy was not found at %CD%\%PY%
    echo Follow "Install" in README.md first ^(it creates the .venv folder^).
    pause
    exit /b 1
)

set "WAIT="
if /i "%~1"=="wait" (
    set "WAIT=1"
    shift
)

set /a TRIES=0
:check
"%PY%" -m studybuddy --check
if not errorlevel 1 goto run

if defined WAIT (
    set /a TRIES+=1
    if %TRIES% lss 40 (
        echo.
        echo Not ready yet. Open LM Studio, load the models and start its server. Trying again in 15 seconds...
        timeout /t 15 /nobreak >nul
        goto check
    )
)

echo.
echo Study Buddy is not ready. Fix the lines marked [FAIL] above:
echo   - open LM Studio, load the three models named above, and turn on its local server
echo   - or correct the setting in the .env file
if defined WAIT goto end
choice /c YN /m "Start Study Buddy anyway"
if errorlevel 2 goto end

:run
echo.
"%PY%" -m studybuddy %1 %2 %3 %4
echo.
echo Study Buddy has stopped ^(exit code %errorlevel%^). If this was not on purpose, the last lines above say why.

:end
pause
endlocal
