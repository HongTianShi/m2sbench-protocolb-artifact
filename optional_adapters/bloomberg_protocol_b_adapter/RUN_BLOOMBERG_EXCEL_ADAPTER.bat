@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_bloomberg_excel_adapter.ps1"
if errorlevel 1 (
  echo.
  echo Excel adapter failed. Please check the export zip or console output.
)
pause
