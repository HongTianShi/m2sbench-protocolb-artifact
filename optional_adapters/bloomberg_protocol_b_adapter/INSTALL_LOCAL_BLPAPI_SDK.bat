@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0INSTALL_LOCAL_BLPAPI_SDK.ps1"
if errorlevel 1 (
  echo.
  echo Local Bloomberg SDK install failed. Please copy the console output or logs folder back to the M2S-Bench machine.
)
pause
