@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_BLOOMBERG_ADAPTER.ps1"
if errorlevel 1 (
  echo.
  echo Adapter failed. Please check the logs folder in this directory.
)
pause
