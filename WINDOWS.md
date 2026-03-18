# Windows Start

## Lokaler Start auf Windows

```powershell
.\run_windows.ps1
```

## Manuell

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-win\Scripts\python.exe .\dash_app.py
```

## Tests

```powershell
.\.venv-win\Scripts\python.exe -m pytest -q
```

## Hinweis

Windows und WSL2 verwenden absichtlich getrennte virtuelle Umgebungen:

- Windows: `.venv-win`
- WSL2: `.venv-wsl`

So vermeiden wir kaputte Interpreterpfade in gemeinsam genutzten Projektordnern.
