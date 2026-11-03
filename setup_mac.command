#!/bin/bash
# One-click setup and launch for macOS. Double-click this file: the first run
# creates .venv and installs requirements.txt, later runs just start NeuroVAT
# (they reinstall only when requirements.txt changes).
cd "$(dirname "$0")" || exit 1

fail() {
  echo "$1"
  read -r -p "Press Return to close."
  exit 1
}

if [ ! -x .venv/bin/python ]; then
  PY=$(command -v python3) && "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' ||
    fail "Python 3.9 or newer is required. Install it from https://www.python.org/downloads/ and run this file again."
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv || fail "Setup failed; see the messages above."
fi

if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing requirements, this takes a few minutes the first time ..."
  .venv/bin/python -m pip install --upgrade pip &&
    .venv/bin/python -m pip install -r requirements.txt ||
    fail "Setup failed; see the messages above."
  cp requirements.txt .venv/requirements.installed
fi

echo "Starting NeuroVAT ..."
.venv/bin/python run_gui.py "$@" || read -r -p "NeuroVAT exited with an error. Press Return to close."
