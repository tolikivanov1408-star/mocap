@echo off
REM Сборка standalone .exe (нужен Visual C++ Redistributable на целевой машине)
setlocal
if not exist .venv\Scripts\python.exe (
  echo Сначала запустите install.bat
  pause
  exit /b 1
)
echo Собираем MoCapStudio.exe ...
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean mocap_studio.spec
if errorlevel 1 (
  echo Ошибка сборки
  pause
  exit /b 1
)
echo.
echo Готово: dist\MoCapStudio.exe
pause
