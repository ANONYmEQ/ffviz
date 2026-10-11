@echo off
rem One-time installation of ffviz (from source). Afterwards use the desktop icon or "Start ffviz.bat".
cd /d "%~dp0"
echo.
echo === ffviz installation ===
echo.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)"
  if errorlevel 1 (
    echo The existing environment uses an unsupported Python version - creating a new one.
    rmdir /s /q ".venv"
  ) else (
    echo Re-using the existing environment .venv
    goto install
  )
)
set "PY="
for %%v in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  echo No suitable Python found ^(3.10 - 3.13^).
  echo Please install Python 3.12 from python.org and run this file again.
  pause
  exit /b 1
)
echo Using %PY%
%PY% -m venv .venv || (echo Could not create the environment & pause & exit /b 1)
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Installation failed & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install -e . || (echo Installation failed & pause & exit /b 1)
echo.
echo Creating desktop shortcut "ffviz" ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\ffviz.lnk'); $s.TargetPath='%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments='-m ffviz.gui'; $s.WorkingDirectory='%~dp0'; $s.Description='ffviz - finite-fault figures'; if (Test-Path '%~dp0packaging\ffviz.ico') { $s.IconLocation='%~dp0packaging\ffviz.ico' }; $s.Save()"
echo.
echo Done! Start ffviz with the desktop icon "ffviz" or "Start ffviz.bat".
pause
