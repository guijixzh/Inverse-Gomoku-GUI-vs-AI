@echo off
rem ============================================================
rem  Antifive (Inverse Gomoku) GUI quick launcher for Windows
rem  Double-click to start the GUI. Flags are forwarded, e.g.:
rem      run_gui.bat --engine heuristic
rem ============================================================
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

set "PY="
rem Optional machine-local override (gitignored): set PY in run_gui.local.bat
if exist "%~dp0run_gui.local.bat" call "%~dp0run_gui.local.bat"
if defined PY goto launch
python -c "import numpy, pygame" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto launch
py -3 -c "import numpy, pygame" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto launch

echo [ERROR] No usable Python found -- numpy and pygame are required.
echo Install Python 3.10+ first, then run: python scripts\setup_env.py --install
echo With conda, activate your environment first, then run this script.
pause
exit /b 1

:launch
%PY% -m antifive %*
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo [ERROR] Launch failed, exit code %RC% -- try: pip install -e ".[gui]"
if not "%RC%"=="0" pause
exit /b %RC%
