@echo off
:: MXG Waveform Designer — Windows quick-start
:: Creates a virtual environment, installs dependencies, and launches the GUI.

setlocal

set "VENV_DIR=%~dp0.venv"
set "PY_SCRIPT=%~dp0mxg_waveform_designer.py"

:: ── 1. Find Python 3.8+ ───────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.8+ from https://python.org and rerun.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

:: ── 2. Create virtual environment (only on first run) ────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
)

:: ── 3. Activate and install / update dependencies ────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

echo Installing / updating dependencies...
pip install --upgrade pip --quiet
pip install -r "%~dp0requirements.txt" --quiet
if errorlevel 1 (
    echo ERROR: Dependency install failed. Check your internet connection.
    pause
    exit /b 1
)

:: ── 4. Launch the GUI ────────────────────────────────────────────────────
echo.
echo Launching MXG Waveform Designer...
python "%PY_SCRIPT%"

endlocal
