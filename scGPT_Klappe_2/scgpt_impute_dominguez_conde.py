"""
scGPT-like masked expression prediction (no scgpt library).

- Loads: ../data/dominguez_conde_immune_tissue_two_donors.h5ad
- Masks 20% of gene expression values per cell
- Tries to load a model from ../models/scGPT (args.json, vocab.json + weights)
  1) TorchScript (torch.jit.load) if available
  2) Otherwise loads a state_dict into a minimal Transformer model (may or may not match your checkpoint)

Run:
  python scgpt_impute.py

Optional:
  python scgpt_impute.py --train  # quick fine-tune (only if state_dict path works)
"""

import os
import json
import glob
import argparse
import random
import math
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

import anndata as ad

# Reduce CPU thread fan-out to avoid memory spikes on large tensors.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dense_row(x_row) -> np.ndarray:
    if hasattr(x_row, "toarray"):
        return np.asarray(x_row.toarray()).reshape(-1)
    return np.asarray(x_row).reshape(-1)


def mask_values(values: np.ndarray, mask_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    values: (G,)
    returns:
      masked_values: (G,) float32
      mask: (G,) bool (True means masked)
    """
    G = values.shape[0]
    m = int(round(G * mask_ratio))
    idx = np.random.choice(G, size=m, replace=False)
    mask = np.zeros(G, dtype=bool)
    mask[idx] = True
    masked = values.copy()
    masked[mask] = 0.0
    return masked.astype(np.float32), mask


def find_weight_files(model_dir: str) -> List[str]:
    pats = ["*.pt", "*.pth", "*.ckpt", "*.bin", "*.jit", "*.ts"]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(os.path.join(model_dir, p)))
    files = [f for f in files if os.path.basename(f) not in ("args.json", "vocab.json")]
    return sorted(files)


def try_load_torchscript(path: str, device: str):
    try:
        return torch.jit.load(path, map_location=device)
    except Exception:
        return None


def try_load_state_dict(path: str):
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
            return obj
        return None
    except Exception:
        return None


# ----------------------------
# Dataset
# ----------------------------
class SingleCellMaskedDataset(Dataset):
    def __init__(
        self,
        adata_obj,
        gene_ids_arr: np.ndarray,
        mask_ratio: float = 0.2,
        add_cls: bool = False,
        cls_id: Optional[int] = None,
        pad_id: int = 0,
        max_len: int = 2048,
        only_nonzero: bool = False,
        flag_normal_id: int = 0,
        flag_masked_id: int = 1,
        flag_pad_id: int = 2,
        flag_cls_id: Optional[int] = None,
    ):
        self.adata = adata_obj
        self.gene_ids_full = gene_ids_arr.astype(np.int64)  # (G,)
        self.mask_ratio = float(mask_ratio)
        self.add_cls = bool(add_cls and (cls_id is not None))
        self.cls_id = cls_id
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.only_nonzero = bool(only_nonzero)
        self.flag_normal_id = int(flag_normal_id)
        self.flag_masked_id = int(flag_masked_id)
        self.flag_pad_id = int(flag_pad_id)
        self.flag_cls_id = flag_cls_id if flag_cls_id is None else int(flag_cls_id)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        row = self.adata.X[idx]
        vals_full = to_dense_row(row).astype(np.float32)  # (G,)

        # --- Top-K Auswahl ---
        if self.only_nonzero:
            nz = np.nonzero(vals_full)[0]
            if nz.size == 0:
                top_idx = np.array([], dtype=np.int64)
            else:
                # Top-K innerhalb nonzero
                k = min(self.max_len, nz.size)
                sub = vals_full[nz]
                part = np.argpartition(sub, -k)[-k:]
                top_idx = nz[part]
        else:
            k = min(self.max_len, vals_full.shape[0])
            top_idx = np.argpartition(vals_full, -k)[-k:]

        # sortiere optional nach absteigender Expression (stabiler)
        if top_idx.size > 0:
            top_idx = top_idx[np.argsort(vals_full[top_idx])[::-1]]

        tokens = self.gene_ids_full[top_idx]               # (L<=K,)
        vals = vals_full[top_idx].astype(np.float32)       # (L<=K,)

        L = tokens.shape[0]
        flags = np.full(self.max_len, self.flag_normal_id, dtype=np.int64)
        # padding auf max_len
        if L < self.max_len:
            pad_n = self.max_len - L
            tokens = np.concatenate([tokens, np.full(pad_n, self.pad_id, dtype=np.int64)], axis=0)
            vals = np.concatenate([vals, np.zeros(pad_n, dtype=np.float32)], axis=0)
            flags[L:] = self.flag_pad_id

        # --- Maskierung nur auf echten (nicht-pad) Positionen ---
        real_len = L
        masked_vals = vals.copy()
        mask = np.zeros(self.max_len, dtype=bool)
        if real_len > 0:
            m = int(round(real_len * self.mask_ratio))
            if m > 0:
                midx = np.random.choice(real_len, size=m, replace=False)
                mask[midx] = True
                masked_vals[midx] = np.float32(0.0)
                flags[midx] = self.flag_masked_id

        # CLS davor (optional)
        if self.add_cls:
            tokens = np.concatenate([[self.cls_id], tokens], axis=0)
            vals = np.concatenate([[np.float32(0.0)], vals], axis=0).astype(np.float32)
            masked_vals = np.concatenate([[np.float32(0.0)], masked_vals], axis=0).astype(np.float32)
            mask = np.concatenate([[False], mask], axis=0)
            cls_flag = self.flag_normal_id if self.flag_cls_id is None else self.flag_cls_id
            flags = np.concatenate([[cls_flag], flags], axis=0)

        return (
            torch.from_numpy(tokens),          # (L,) long
            torch.from_numpy(masked_vals),     # (L,) float
            torch.from_numpy(vals),            # (L,) float
            torch.from_numpy(mask),            # (L,) bool
            torch.from_numpy(flags),           # (L,) long
        )


def collate_batch(batch):
    gene_ids = torch.stack([b[0] for b in batch], dim=0)
    inp = torch.stack([b[1] for b in batch], dim=0)
    tgt = torch.stack([b[2] for b in batch], dim=0)
    m = torch.stack([b[3] for b in batch], dim=0)
    flags = torch.stack([b[4] for b in batch], dim=0)
    return gene_ids, inp, tgt, m, flags


# ----------------------------
# Minimal Transformer model
# ----------------------------
def infer_n_heads_from_args(args_dict: dict) -> int:
    return int(args_dict.get("n_heads", args_dict.get("nhead", args_dict.get("num_heads", 8))))


class ScGPTSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(out)


class ScGPTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.self_attn = ScGPTSelfAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, key_padding_mask=key_padding_mask)
        h = self.norm2(x)
        h = self.linear2(F.gelu(self.linear1(h)))
        x = x + h
        return x


class ScGPTCompatModel(nn.Module):
    """
    Passende Namespaces für den Checkpoint:
      encoder.embedding.*
      encoder.enc_norm.*
      flag_encoder.*
      value_encoder.linear1/linear2/norm.*
      transformer_encoder.layers.*
      decoder.fc.*
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        n_flags: int,
        pad_id: int,
        decoder_h1: Optional[int] = None,
        decoder_h2: Optional[int] = None,
        decoder_out: int = 1,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.encoder = nn.Module()
        self.encoder.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        self.encoder.enc_norm = nn.LayerNorm(d_model)

        self.flag_encoder = nn.Embedding(n_flags, d_model)

        self.value_encoder = nn.Module()
        self.value_encoder.linear1 = nn.Linear(1, d_model, bias=True)
        self.value_encoder.linear2 = nn.Linear(d_model, d_model, bias=True)
        self.value_encoder.norm = nn.LayerNorm(d_model)

        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList(
            [ScGPTEncoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers)]
        )
        h1 = d_model if decoder_h1 is None else int(decoder_h1)
        h2 = d_model if decoder_h2 is None else int(decoder_h2)
        self.decoder = nn.Module()
        self.decoder.fc = nn.Sequential(
            nn.Linear(d_model, h1, bias=True),
            nn.GELU(),
            nn.Linear(h1, h2, bias=True),
            nn.GELU(),
            nn.Linear(h2, int(decoder_out), bias=True),
        )

    def forward(self, gene_ids: torch.Tensor, values: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        gene_ids = gene_ids.long()
        flags = flags.long()
        values = values.to(dtype=self.encoder.embedding.weight.dtype)
        x_gene = self.encoder.embedding(gene_ids)
        x_gene = self.encoder.enc_norm(x_gene)

        x_val = self.value_encoder.linear2(F.gelu(self.value_encoder.linear1(values.unsqueeze(-1))))
        x_val = self.value_encoder.norm(x_val)
        x_flag = self.flag_encoder(flags)

        x = x_gene + x_val + x_flag
        key_padding_mask = (gene_ids == self.pad_id)
        for layer in self.transformer_encoder.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        y = self.decoder.fc(x).squeeze(-1)
        return y


def infer_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int, int, int, int, int]:
    d_model = int(sd["encoder.embedding.weight"].shape[1])
    vocab_size = int(sd["encoder.embedding.weight"].shape[0])
    n_flags = int(sd["flag_encoder.weight"].shape[0])

    layer_ids = []
    for k in sd.keys():
        if k.startswith("transformer_encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.append(int(parts[2]))
    n_layers = (max(layer_ids) + 1) if layer_ids else 0

    d_ff = int(sd["transformer_encoder.layers.0.linear1.weight"].shape[0])

    dec0_out = int(sd["decoder.fc.0.weight"].shape[0])
    dec2_out = int(sd["decoder.fc.2.weight"].shape[0])
    dec4_out = int(sd["decoder.fc.4.weight"].shape[0])
    return vocab_size, d_model, d_ff, n_layers, n_flags, dec0_out, dec2_out, dec4_out


# ----------------------------
# Eval / Train
# ----------------------------
@torch.no_grad()
def evaluate(dataloader, model, device: str, use_torchscript: bool, show_progress: bool = True) -> Tuple[float, float]:
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    total_n = 0

    it = tqdm(dataloader, desc="Eval", leave=False, disable=not show_progress)
    for gene_ids, inp, tgt, m, flags in it:
        gene_ids = gene_ids.to(device)
        inp = inp.to(device).float()
        tgt = tgt.to(device).float()
        mask = m.to(device).bool()
        flags = flags.to(device)

        if use_torchscript:
            try:
                pred = model(gene_ids, inp, flags)
            except TypeError:
                pred = model(gene_ids, inp)
        else:
            pred = model(gene_ids, inp, flags)
        n = int(mask.sum().item())
        if n == 0:
            continue

        diff = pred[mask] - tgt[mask]
        total_se += float((diff ** 2).sum().item())
        total_ae += float(diff.abs().sum().item())
        total_n += n

    mse = total_se / max(total_n, 1)
    mae = total_ae / max(total_n, 1)
    return mse, mae


def train_one_epoch(dataloader, model, optim, device: str, show_progress: bool = True) -> float:
    model.train()
    total = 0.0
    total_n = 0

    it = tqdm(dataloader, desc="Train", leave=False, disable=not show_progress)
    for gene_ids, inp, tgt, m, flags in it:
        gene_ids = gene_ids.to(device)
        inp = inp.to(device).float()
        tgt = tgt.to(device).float()
        mask = m.to(device).bool()
        flags = flags.to(device)

        pred = model(gene_ids, inp, flags)
        if int(mask.sum().item()) == 0:
            continue

        loss = F.mse_loss(pred[mask], tgt[mask])

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total += float(loss.item()) * int(mask.sum().item())
        total_n += int(mask.sum().item())

    return total / max(total_n, 1)


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data/dominguez_conde_immune_tissue_two_donors.h5ad")
    p.add_argument("--model_dir", default="./models/scGPT")
    p.add_argument("--mask_ratio", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cls", action="store_true", help="kein CLS token prepending")
    p.add_argument("--train", action="store_true", help="kurzes Fine-tuning (nur wenn state_dict kompatibel ist)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=2048, help="Top-K Gene pro Zelle (Sequenzlänge ohne CLS)")
    p.add_argument("--only_nonzero", action="store_true", help="Top-K nur aus nonzero Genen")
    p.add_argument("--bf16_cpu", action="store_true", help="Inference/Eval in bfloat16 auf CPU")
    p.add_argument("--no_progress", action="store_true", help="Deaktiviert tqdm Fortschrittsanzeige")
    args_cli = p.parse_args()

    set_seed(args_cli.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args_path = os.path.join(args_cli.model_dir, "args.json")
    vocab_path = os.path.join(args_cli.model_dir, "vocab.json")
    if not os.path.exists(args_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError("args.json oder vocab.json fehlt im model_dir")

    with open(args_path, "r", encoding="utf-8") as f:
        model_args = json.load(f)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # vocab parsing
    gene_to_id: Dict[str, int] = {}
    if isinstance(vocab, dict):
        if all(isinstance(v, int) for v in vocab.values()):
            gene_to_id = {k: int(v) for k, v in vocab.items()}
        elif "stoi" in vocab and isinstance(vocab["stoi"], dict):
            gene_to_id = {k: int(v) for k, v in vocab["stoi"].items()}
        elif "token_to_id" in vocab and isinstance(vocab["token_to_id"], dict):
            gene_to_id = {k: int(v) for k, v in vocab["token_to_id"].items()}
        else:
            raise ValueError("Unbekanntes vocab.json Format (erwartet gene->id oder stoi/token_to_id).")
    else:
        raise ValueError("Unbekanntes vocab.json Format (kein dict).")

    pad_token = model_args.get("pad_token", "<pad>")
    cls_token = model_args.get("cls_token", "<cls>")
    mask_token = model_args.get("mask_token", "<mask>")

    pad_id = gene_to_id.get(pad_token, 0)
    cls_id = gene_to_id.get(cls_token, None)
    mask_id = gene_to_id.get(mask_token, None)
    print(f"Device: {device}")
    print(f"pad_id={pad_id} cls_id={cls_id} mask_id={mask_id}")

    # Load data
    if not os.path.exists(args_cli.data):
        raise FileNotFoundError(f"Dataset nicht gefunden: {args_cli.data}")
    adata = ad.read_h5ad(args_cli.data)

    var_names = np.array(adata.var_names).astype(str)
    gene_ids = np.array([gene_to_id.get(g, -1) for g in var_names], dtype=np.int64)
    keep = gene_ids >= 0
    print(f"Genes kept: {int(keep.sum())} / {len(keep)} (dropped {int((~keep).sum())})")

    adata_f = adata[:, keep].copy()
    gene_ids_f = gene_ids[keep]

    add_cls = (not args_cli.no_cls) and (cls_id is not None)
    vocab_size = int(max(gene_to_id.values()) + 1)
    n_heads = infer_n_heads_from_args(model_args)
    d_model_args = int(model_args.get("embsize", model_args.get("d_model", 512)))
    n_layers_args = int(model_args.get("n_layers", model_args.get("nlayers", model_args.get("num_layers", 6))))
    d_ff_args = int(model_args.get("d_hid", model_args.get("dim_feedforward", model_args.get("d_ff", 4 * d_model_args))))
    n_flags_args = int(model_args.get("n_flags", model_args.get("num_flags", 4)))
    model = ScGPTCompatModel(
        vocab_size=vocab_size,
        d_model=d_model_args,
        d_ff=d_ff_args,
        n_heads=n_heads,
        n_layers=n_layers_args,
        n_flags=n_flags_args,
        pad_id=pad_id,
    ).to(device)
    n_flags_active = n_flags_args

    # Load weights
    weight_files = find_weight_files(args_cli.model_dir)
    if not weight_files:
        print("WARN: Keine Gewichtsdatei im model_dir gefunden. Modell bleibt zufällig initialisiert.")
        use_ts = False
    else:
        torchscript_model = None
        loaded_path = None

        for wf in weight_files:
            ts = try_load_torchscript(wf, device=device)
            if ts is not None:
                torchscript_model = ts
                loaded_path = wf
                break

        if torchscript_model is not None:
            model = torchscript_model  # replace
            use_ts = True
            print(f"TorchScript geladen: {loaded_path}")
        else:
            use_ts = False
            loaded = False
            for wf in weight_files:
                sd = try_load_state_dict(wf)
                if sd is not None and "encoder.embedding.weight" in sd:
                    sd_filtered = {k: v for k, v in sd.items() if not k.startswith("mvc_decoder.")}
                    (
                        vocab_size_sd,
                        d_model_sd,
                        d_ff_sd,
                        n_layers_sd,
                        n_flags_sd,
                        dec0,
                        dec2,
                        dec4,
                    ) = infer_from_state_dict(sd_filtered)
                    model = ScGPTCompatModel(
                        vocab_size=vocab_size_sd,
                        d_model=d_model_sd,
                        d_ff=d_ff_sd,
                        n_heads=n_heads,
                        n_layers=n_layers_sd,
                        n_flags=n_flags_sd,
                        pad_id=pad_id,
                        decoder_h1=dec0,
                        decoder_h2=dec2,
                        decoder_out=dec4,
                    ).to(device)
                    n_flags_active = n_flags_sd
                    try:
                        model.load_state_dict(sd_filtered, strict=True)
                        missing, unexpected = [], []
                        print("state_dict strict=True erfolgreich (ohne mvc_decoder.*).")
                    except RuntimeError as e:
                        print("state_dict strict=True fehlgeschlagen; fallback strict=False.")
                        print(f"strict Fehler: {e}")
                        missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
                    loaded = True
                    loaded_path = wf
                    print(f"state_dict geladen: {loaded_path}")
                    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
                    break
            if not loaded:
                print("WARN: Gewichte gefunden, aber kein kompatibles Format. Modell bleibt zufällig initialisiert.")

    # Dataset bauen, nachdem n_flags aus Config/Checkpoint feststeht
    # Map flags safely to available embedding rows.
    flag_normal_id = 0
    if n_flags_active <= 1:
        flag_masked_id = 0
        flag_pad_id = 0
        cls_flag_id = 0 if add_cls else None
    elif n_flags_active == 2:
        flag_masked_id = 1
        flag_pad_id = 0
        cls_flag_id = 0 if add_cls else None
    elif n_flags_active == 3:
        flag_masked_id = 1
        flag_pad_id = 2
        cls_flag_id = 0 if add_cls else None
    else:
        flag_masked_id = 1
        flag_pad_id = 2
        cls_flag_id = 3 if add_cls else None

    if add_cls and (cls_flag_id is None or cls_flag_id >= n_flags_active):
        print("WARN: add_cls aktiv, aber kein separater CLS-Flag verfügbar. CLS wird auf normal gemappt.")
    print(
        f"Flag mapping: normal={flag_normal_id}, masked={flag_masked_id}, "
        f"pad={flag_pad_id}, cls={cls_flag_id}, n_flags={n_flags_active}"
    )
    ds = SingleCellMaskedDataset(
        adata_f,
        gene_ids_f,
        mask_ratio=args_cli.mask_ratio,
        add_cls=add_cls,
        cls_id=cls_id,
        pad_id=pad_id,
        max_len=args_cli.max_len,
        only_nonzero=args_cli.only_nonzero,
        flag_normal_id=flag_normal_id,
        flag_masked_id=flag_masked_id,
        flag_pad_id=flag_pad_id,
        flag_cls_id=cls_flag_id,
    )
    dl = DataLoader(ds, batch_size=args_cli.batch_size, shuffle=True, num_workers=args_cli.num_workers, collate_fn=collate_batch)

    # Debug sanity-check: effective sequence length and padding usage.
    gene_ids_dbg, _, _, _, _ = next(iter(dl))
    print("Debug batch shape (B,L):", tuple(gene_ids_dbg.shape))
    print("Debug pad fraction:", float((gene_ids_dbg == pad_id).float().mean().item()))

    if args_cli.bf16_cpu and device == "cpu":
        model = model.to(dtype=torch.bfloat16)
        print("CPU bf16 enabled for model.")

    # Eval before training
    mse, mae = evaluate(dl, model, device=device, use_torchscript=use_ts, show_progress=not args_cli.no_progress)
    print(f"Eval (masked-only): MSE={mse:.6f}  MAE={mae:.6f}")

    # Optional train
    if args_cli.train:
        if use_ts:
            raise RuntimeError("Training mit TorchScript nicht unterstützt. Nutze state_dict-Checkpoint.")
        optim = torch.optim.AdamW(model.parameters(), lr=args_cli.lr, weight_decay=0.01)
        for e in range(args_cli.epochs):
            loss = train_one_epoch(dl, model, optim, device=device, show_progress=not args_cli.no_progress)
            mse, mae = evaluate(dl, model, device=device, use_torchscript=False, show_progress=not args_cli.no_progress)
            print(f"Epoch {e+1}/{args_cli.epochs}: train_loss={loss:.6f}  eval_mse={mse:.6f}  eval_mae={mae:.6f}")

    # Example prediction on one batch: fill masked values with predictions
    gene_ids_b, inp_b, tgt_b, m_b, flags_b = next(iter(dl))
    gene_ids_b = gene_ids_b.to(device)
    inp_dtype = torch.bfloat16 if (args_cli.bf16_cpu and device == "cpu") else torch.float32
    inp_b = inp_b.to(device, dtype=inp_dtype)
    flags_b = flags_b.to(device)
    with torch.no_grad():
        if use_ts:
            try:
                pred_b = model(gene_ids_b, inp_b, flags_b).float().detach().cpu().numpy()
            except TypeError:
                pred_b = model(gene_ids_b, inp_b).float().detach().cpu().numpy()
        else:
            pred_b = model(gene_ids_b, inp_b, flags_b).float().detach().cpu().numpy()

    inp_np = inp_b.detach().cpu().numpy()
    mask_np = m_b.numpy().astype(bool)
    recon = inp_np.copy()
    recon[mask_np] = pred_b[mask_np]

    print("Example batch reconstruction:")
    print("  inp shape:", inp_np.shape, "pred shape:", pred_b.shape, "recon shape:", recon.shape)
    print("  masked fraction in batch:", mask_np.mean())

    # Save recon example
    out_path = os.path.join(args_cli.model_dir, "recon_example.npy")
    np.save(out_path, recon)
    print("Saved example recon to:", out_path)


if __name__ == "__main__":
    main()
