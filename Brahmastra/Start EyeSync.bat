@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo EyeSync's local Python environment is missing. See README.md for setup.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "dashboard_launcher.py" %*
