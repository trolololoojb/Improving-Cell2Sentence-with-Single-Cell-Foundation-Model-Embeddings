# C2S Donor Inference (Neuer Ansatz)

Dieses Verzeichnis enthält einen **eigenständigen** Notebook-Workflow (ohne Projektcode), um mit dem vortrainierten Modell
`vandijklab/C2S-Pythia-410m-cell-type-prediction` Zelltypen für ein Donor-Subset vorherzusagen.

## Dateien
- `c2s_donor_celltype_prediction.ipynb`: End-to-end Inferenz

## Start
1. Optional venv aktivieren:
   - `source .venv/bin/activate`
2. Jupyter starten:
   - `jupyter notebook`
3. Notebook öffnen und Zellen der Reihe nach ausführen.

Hinweis: Im Notebook ist standardmäßig `DONOR_COLUMN='batch_condition'` und `DONOR_VALUE='A29'` gesetzt.
