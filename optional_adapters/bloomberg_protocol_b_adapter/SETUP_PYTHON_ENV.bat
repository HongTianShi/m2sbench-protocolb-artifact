@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0SETUP_PYTHON_ENV.ps1"
if errorlevel 1 (
  echo.
  echo Python setup failed. Please copy the console output back to the M2S-Bench machine.
)
pause
