@echo off
rem Builds dist\ffviz\ffviz.exe and dist\ffviz-windows.zip (run install.bat first).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install "pyinstaller>=6.3" || (pause & exit /b 1)
".venv\Scripts\python.exe" -m PyInstaller packaging\ffviz.spec --noconfirm --clean || (echo Build failed & pause & exit /b 1)
copy /y README.md dist\ffviz\README.md >nul
copy /y LICENSE dist\ffviz\LICENSE.txt >nul
copy /y THIRD_PARTY_NOTICES.txt dist\ffviz\THIRD_PARTY_NOTICES.txt >nul
powershell -NoProfile -Command "Compress-Archive -Path 'dist\ffviz' -DestinationPath 'dist\ffviz-windows.zip' -Force"
echo.
echo Done:  dist\ffviz\ffviz.exe   and   dist\ffviz-windows.zip
pause
