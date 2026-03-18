#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x ".venv-wsl/bin/python" ]]; then
  python3 -m venv .venv-wsl
fi

source .venv-wsl/bin/activate
python -m pip install -r requirements.txt
exec python dash_app.py
