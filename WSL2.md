# WSL2 Start

## Empfehlung

Das Projekt laeuft unter WSL2 am stabilsten und schnellsten, wenn es im Linux-Dateisystem liegt und nicht unter `/mnt/c/...` oder in OneDrive.

## Start in WSL2

```bash
cd /pfad/zum/projekt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python dash_app.py
```

Danach im Browser:

`http://localhost:8053`

## Komfort-Start

Alternativ:

```bash
bash run_wsl.sh
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## VS Code

Wenn Du das Projekt ueber die VS-Code-WSL-Erweiterung oeffnest, waehle als Interpreter die WSL-Umgebung `.venv/bin/python`.
