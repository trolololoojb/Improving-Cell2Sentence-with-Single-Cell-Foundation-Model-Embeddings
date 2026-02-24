import os
import json
import random
from datetime import datetime

import numpy as np
import anndata
import scanpy as sc

import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data
from transformers import TrainingArguments


# -------------------------
# Config
# -------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

H5AD_PATH = "dominguez_conde_immune_tissue_two_donors (1).h5ad"

# Output
WORKDIR = "./c2s_runs"
RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTDIR = os.path.join(WORKDIR, RUN_NAME)
os.makedirs(OUTDIR, exist_ok=True)

ARROW_SAVE_DIR = os.path.join(OUTDIR, "csdata")
MODEL_SAVE_DIR = os.path.join(OUTDIR, "csmodel")
os.makedirs(ARROW_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model (pretrained)
BASE_MODEL = "vandijklab/C2S-Pythia-410m-cell-type-prediction"  # guter Start für Cell-Type Prediction

# Task
TRAINING_TASK = "cell_type_prediction"

# Cell sentence length
TOP_K_GENES = 200  # häufig in C2S Setups

# -------------------------
# Helpers
# -------------------------
def preprocess_adata(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Solider Default: QC + Normalize + log1p (base 10).
    Wenn dein .h5ad schon preprocessed ist, kannst du das vereinfachen.
    """
    adata = adata.copy()

    # Ensure unique gene names
    adata.var_names_make_unique()

    # Basic QC (optional, aber sinnvoll)
    # Mito-Flag (human/mouse üblich: "MT-")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Light filtering (nicht zu aggressiv, damit Labels nicht kaputt gehen)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize total counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # log1p base 10 (Cell2Sentence weicht hier gerne vom Standard-ln ab)
    # scanpy kann base nicht direkt, daher per numpy:
    # log10(1 + x) = ln(1+x)/ln(10)
    X = adata.X
    # sparse safe
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.copy()
        X.data = np.log1p(X.data) / np.log(10.0)
        adata.X = X
    else:
        adata.X = np.log1p(X) / np.log(10.0)

    return adata


def safe_label_cols(adata: anndata.AnnData):
    # Welche obs-Spalten speichern wir mit?
    wanted = ["cell_type", "tissue", "batch_condition", "organism", "sex"]
    return [c for c in wanted if c in adata.obs.columns]


# -------------------------
# Main
# -------------------------
def main():
    print(f"[1/5] Load h5ad: {H5AD_PATH}")
    adata = anndata.read_h5ad(H5AD_PATH)

    if "cell_type" not in adata.obs.columns:
        raise ValueError("adata.obs muss eine Spalte 'cell_type' enthalten (Ground Truth Label).")

    # Optional: Wenn ihr sicher seid, dass es schon preprocessed ist, kommentiert das aus
    print("[2/5] Preprocess")
    adata = preprocess_adata(adata)

    label_cols = safe_label_cols(adata)
    print(f"Label columns saved into dataset: {label_cols}")

    # ---- Build Arrow dataset + vocab
    print("[3/5] Convert AnnData -> Arrow dataset (cell sentences) + vocabulary")
    arrow_ds, vocab = cs.CSData.adata_to_arrow(
        adata,
        random_state=SEED,
        sentence_delimiter=" ",
        label_col_names=label_cols,
    )

    # ---- Split
    print("[3.5/5] Split Arrow dataset into train/val/test (80/10/10)")
    ds_splits, split_indices = cs.utils.train_test_split_arrow_ds(arrow_ds)

    # Save indices for reproducibility
    split_path = os.path.join(OUTDIR, "split_indices.json")
    with open(split_path, "w") as f:
        json.dump(split_indices, f, indent=2)
    print(f"Saved split indices to: {split_path}")

    # ---- Save CSData to disk
    print("[4/5] Save CSData to disk")
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocab,
        save_dir=ARROW_SAVE_DIR,
        save_name="dataset_arrow",
        dataset_backend="arrow",
    )
    print(csdata)  # debug summary

    # ---- Load model wrapper
    print("[4.2/5] Init CSModel")
    csmodel = cs.CSModel(
        model_name_or_path=BASE_MODEL,
        save_dir=MODEL_SAVE_DIR,
        save_name="finetuned_cell_type_pred",
    )

    # ---- Training args (solider Default)
    datetimestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    hf_outdir = os.path.join(MODEL_SAVE_DIR, f"{datetimestamp}_finetune_{TRAINING_TASK}")
    os.makedirs(hf_outdir, exist_ok=True)

    train_args = TrainingArguments(
        bf16=True,                 # falls GPU bf16 kann; sonst: fp16=True oder beides False
        fp16=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch 32
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=5,
        logging_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        output_dir=hf_outdir,
        report_to="none",
    )

    # ---- Fine-tune
    print("[4.6/5] Fine-tune")
    csmodel.fine_tune(
        csdata=csdata,
        task=TRAINING_TASK,
        train_args=train_args,
        loss_on_response_only=False,   # wie im C2S Tutorial
        top_k_genes=TOP_K_GENES,
        max_eval_samples=500,
        data_split_indices_dict={
            "train": split_indices["train"],
            "val": split_indices["val"],
            "test": split_indices.get("test", []),
        },
    )
    print(f"Finetuned model saved under: {hf_outdir}")

    # ---- Inference on test split + simple accuracy
    print("[5/5] Inference on test split + Accuracy")

    # Wir nutzen hier direkt den Arrow Datensatz und selektieren Test-Indizes
    test_ids = split_indices.get("test", [])
    if len(test_ids) == 0:
        print("No test split found. Skipping evaluation.")
        return

    # Erzeuge ein CSData-Objekt für Test (einfach: neues Arrow-Subset speichern)
    test_arrow = arrow_ds.select(test_ids)

    csdata_test = cs.CSData.csdata_from_arrow(
        arrow_dataset=test_arrow,
        vocabulary=vocab,
        save_dir=ARROW_SAVE_DIR,
        save_name="dataset_arrow_test",
        dataset_backend="arrow",
    )

    preds = predict_cell_types_of_data(
        csdata=csdata_test,
        csmodel=csmodel,
        n_genes=TOP_K_GENES,
        max_new_tokens=32,
    )

    # Ground truth
    # Im Arrow sind label cols gespeichert; meist heißt die Spalte exakt wie obs ("cell_type").
    y_true = [test_arrow[i]["cell_type"] for i in range(len(test_arrow))]
    y_pred = [p.strip() for p in preds]

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    print(f"Test accuracy (exact match): {acc:.4f}")

    # Save predictions
    pred_path = os.path.join(OUTDIR, "test_predictions.jsonl")
    with open(pred_path, "w") as f:
        for a, b, c in zip(y_true, y_pred, [test_arrow[i]["cell_sentence"] for i in range(len(test_arrow))]):
            f.write(json.dumps({"y_true": a, "y_pred": b, "cell_sentence": c}) + "\n")
    print(f"Saved test predictions to: {pred_path}")


if __name__ == "__main__":
    main()