@echo off
REM Установка MoCap Studio на Windows
setlocal

where py >nul 2>nul
if errorlevel 1 (
  echo.
  echo Нужен Python 3.10-3.12: https://www.python.org/downloads/
  echo При установке отметьте галочку "Add Python to PATH".
  echo.
  pause
  exit /b 1
)

if not exist .venv (
  echo [1/3] Создаём виртуальное окружение...
  py -3.11 -m venv .venv 2>nul
  if not exist .venv\Scripts\python.exe py -3 -m venv .venv
)

if not exist .venv\Scripts\python.exe (
  echo.
  echo Не удалось создать окружение. Проверьте, что Python установлен и доступен.
  pause
  exit /b 1
)

echo [2/3] Ставим зависимости (numpy, scipy, opencv, PySide6, mediapipe)...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo.
  echo Ошибка установки pip. Проверьте интернет и повторите.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Ошибка установки зависимостей. Проверьте интернет и повторите.
  pause
  exit /b 1
)

echo [3/3] Скачиваем модели распознавания (~38 МБ)...
.venv\Scripts\python.exe tools\download_models.py
if errorlevel 1 (
  echo.
  echo Не удалось скачать модели - программа попробует сделать это при первом запуске.
)

echo.
echo Готово. Запуск: run.bat
pause
