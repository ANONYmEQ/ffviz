@echo off
rem Starts the ffviz app (no console window)
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo ffviz is not installed yet.
  echo Please double-click "install.bat" first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m ffviz.gui
