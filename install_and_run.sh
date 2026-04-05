#!/usr/bin/env bash
# MXG Waveform Designer — macOS / Linux quick-start
# Creates a virtual environment, installs dependencies, and launches the GUI.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PY_SCRIPT="$SCRIPT_DIR/mxg_waveform_designer.py"

# ── 1. Find Python 3.8+ ───────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Install Python 3.8+ and rerun."
    exit 1
fi

echo "Found $($PYTHON --version)"

# ── 2. Create virtual environment (only on first run) ────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi

# ── 3. Activate and install / update dependencies ─────────────────────────
source "$VENV_DIR/bin/activate"

echo "Installing / updating dependencies..."
pip install --upgrade pip --quiet
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

# ── 4. macOS: ensure tkinter is available ─────────────────────────────────
# tkinter ships with the python.org installer but NOT with Homebrew python.
# If missing, install python-tk: brew install python-tk  (macOS)
#                                 sudo apt install python3-tk  (Debian/Ubuntu)
python -c "import tkinter" 2>/dev/null || {
    echo ""
    echo "WARNING: tkinter is not available in this Python."
    echo "  macOS:         brew install python-tk"
    echo "  Debian/Ubuntu: sudo apt install python3-tk"
    echo ""
}

# ── 5. Launch the GUI ─────────────────────────────────────────────────────
echo ""
echo "Launching MXG Waveform Designer..."
python "$PY_SCRIPT"
