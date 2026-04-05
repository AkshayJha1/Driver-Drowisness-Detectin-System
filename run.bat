@echo off
cd /d "%~dp0"
echo Running Driver Drowsiness Detection...
python main.py
if errorlevel 1 (
  echo.
  echo If python was not found, try: py main.py
  echo Or install Python from https://www.python.org/downloads/ and tick "Add Python to PATH"
  pause
)
