@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 app.py --open-browser
) else (
  python app.py --open-browser
)
