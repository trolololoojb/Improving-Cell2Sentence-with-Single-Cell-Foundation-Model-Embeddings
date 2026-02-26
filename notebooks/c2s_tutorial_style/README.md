# C2S Tutorial-Style Notebooks

Diese beiden Notebooks sind im Stil der offiziellen Cell2Sentence-Tutorials aufgebaut:
- Tutorial 3: Finetuning on new datasets
- Tutorial 4: Cell type prediction

## Reihenfolge
1. `1_c2s_finetune_new_dataset.ipynb`
2. `2_c2s_cell_type_prediction.ipynb`

## Hinweise
- Notebook 1 speichert `run_info.json` und `split_indices.json` in `./runs/<timestamp>/`.
- In Notebook 2 muss `RUN_INFO_PATH` auf diese `run_info.json` gesetzt werden.
- Prediction nutzt `max_num_tokens` (nicht `max_new_tokens`), um den bekannten Konflikt mit `cell2sentence` zu vermeiden.
