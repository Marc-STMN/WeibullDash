$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".venv-win\\Scripts\\python.exe")) {
    python -m venv .venv-win
}

& ".venv-win\\Scripts\\python.exe" -m pip install -r requirements.txt
& ".venv-win\\Scripts\\python.exe" .\dash_app.py

