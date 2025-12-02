# Weibull Dash App

Dash-basierte Version des Weibull Analysis Tools. Excel per Drag & Drop hochladen, Analyse starten und Ergebnisse als ZIP herunterladen oder serverseitig speichern.

## Installation
```
cd my-streamlit-app
pip install -r requirements.txt
```

## Start (Port 8053)
```
python dash_app.py
```
Anschliessend im Browser `http://localhost:8053` öffnen.

## Bedienung
- Excel-Datei auf die Upload-Fläche ziehen oder wählen (Sheet `Ergebnisse` und `Parameter` wie gewohnt).
- Parameter-Schlüssel (Auftrags-Nr. oder Werkstoff) und Konfidenzniveau wählen.
- Optional Kommentar und Zielwert für die Ausfallwahrscheinlichkeit eingeben.
- "Analyse starten" klicken, Plot und Kennwerte erscheinen.
- "Ergebnis herunterladen" speichert ein ZIP (Plot + JSON + CSV). Browser fragt nach Zielordner.
- Optional kann zusätzlich ein Serverpfad angegeben werden (Standard: ./exports innerhalb des Projekts).

## Tests
```
pytest
```
