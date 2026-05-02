#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh  –  One-shot environment bootstrap for Prompt Injection Detector
#
# Run once:  chmod +x setup_env.sh && ./setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON_TARGET="python3.11"   # 3.11 = safest for torch + transformers + langkit
VENV_DIR=".venv"

echo "──────────────────────────────────────────────"
echo " Prompt Injection Detector – Environment Setup"
echo "──────────────────────────────────────────────"

# ── 1. Verify Python version ──────────────────────────────────────────────────
if ! command -v "$PYTHON_TARGET" &>/dev/null; then
  echo ""
  echo "[!] $PYTHON_TARGET not found. Options to install it:"
  echo "    macOS (pyenv): pyenv install 3.11.9 && pyenv local 3.11.9"
  echo "    macOS (brew):  brew install python@3.11"
  echo "    conda:         conda create -n pid python=3.11 && conda activate pid"
  echo ""
  echo "    Alternatively, if python3.12 is available it will also work."
  echo "    Python 3.13 is NOT recommended – torch wheels may be missing."
  exit 1
fi

PY_VER=$("$PYTHON_TARGET" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[✓] Found $PYTHON_TARGET  (v$PY_VER)"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[*] Creating virtual environment at $VENV_DIR …"
  "$PYTHON_TARGET" -m venv "$VENV_DIR"
else
  echo "[✓] Virtual environment already exists at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "[✓] Activated: $(python --version)"

# ── 3. Upgrade pip / setuptools ──────────────────────────────────────────────
echo "[*] Upgrading pip …"
pip install --quiet --upgrade pip setuptools wheel

# ── 4. Install dependencies ───────────────────────────────────────────────────
echo "[*] Installing requirements (first run downloads ~400 MB of ML models) …"
pip install --quiet -r requirements.txt

echo ""
echo "──────────────────────────────────────────────"
echo " Setup complete!  Activate with:"
echo "   source $VENV_DIR/bin/activate"
echo " Then run the diagnostic:"
echo "   python detector.py"
echo "──────────────────────────────────────────────"
