#!/usr/bin/env bash
# Build a desktop bundle (macOS .app / Windows / Linux folder) with PyInstaller.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --quiet pyinstaller
python scripts/make_icon.py
pyinstaller packaging/ai_made_easy.spec --noconfirm --clean
echo "Bundle written to dist/"
