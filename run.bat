@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Сначала запустите install.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_gui.py %*
