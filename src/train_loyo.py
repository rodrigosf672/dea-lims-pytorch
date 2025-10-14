import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

from .model import MLP, MLPConfig

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def make_features(
    df: pd.DataFrame,
    use_state: bool = True,
    use_geo: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return X, y, and a metadata dict with encoder and feature names."""
    y = df["purity_idx"].values.astype(int)

    num_cols = ["month_sin","month_cos"]
    geo_cols = ["is_border_state","is_neighbor_of_border","border_tier"] if use_geo else []
    cat_cols = []
    if use_state:
        cat_cols.append("State")
    cat_cols += ["Drug_Type","Net_Weight","Price"]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(df[cat_cols].fillna("NA")) if cat_cols else np.zeros((len(df),0))
    X_num = df[num_cols + geo_cols].fillna(0).to_numpy(dtype=float)

    X = np.hstack([X_num, X_cat]).astype(np.float32)
    meta = {
        "encoder": enc,
        "feature_names": (num_cols + geo_cols) + list(enc.get_feature_names_out(cat_cols)) if cat_cols else (num_cols + geo_cols),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "geo_cols": geo_cols,
    }
    return X, y, meta

class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def class_weights_from_labels(y, num_classes=10):
    counts = np.bincount(y, minlength=num_classes).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    w = inv / inv.sum() * num_classes  # normalize around 1.0
    return w

def train_one_fold(Xtr, ytr, Xva, yva, in_dim, out_dim=10, lr=1e-3, epochs=50, bs=2048, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(MLPConfig(in_dim=in_dim, out_dim=out_dim)).to(device)
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        cw = None
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    train_ds = NPDataset(Xtr, ytr)
    val_ds = NPDataset(Xva, yva)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs*2, shuffle=False, num_workers=0)

    best_acc, best_state, patience, patience_lim = 0.0, None, 0, 7
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # validate on "val" (here we use test fold as val to drive early stop)
        model.eval()
        allp, ally = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                logits = model(xb)
                allp.append(logits.cpu().numpy())
                ally.append(yb.numpy())
        preds = np.argmax(np.vstack(allp), axis=1)
        ytrue = np.hstack(ally)
        acc = accuracy_score(ytrue, preds)
        sched.step(1-acc)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.cpu() if hasattr(v, "device") else v for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= patience_lim:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def loyo(df: pd.DataFrame, out_dir: Path, use_state=True, use_geo=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    rows = []
    for heldout in years:
        tr = df[df["Year"] != heldout].reset_index(drop=True)
        te = df[df["Year"] == heldout].reset_index(drop=True)

        Xtr, ytr, meta = make_features(tr, use_state=use_state, use_geo=use_geo)
        enc = meta["encoder"]; num_cols = meta["num_cols"]; geo_cols = meta["geo_cols"]; cat_cols = meta["cat_cols"]

        if len(cat_cols):
            Xte_cat = enc.transform(te[cat_cols].fillna("NA"))
        else:
            Xte_cat = np.zeros((len(te),0))
        Xte_num = te[num_cols + geo_cols].fillna(0).to_numpy(dtype=float)
        Xte = np.hstack([Xte_num, Xte_cat]).astype(np.float32)
        yte = te["purity_idx"].values.astype(int)

        cw = class_weights_from_labels(ytr, num_classes=10)
        model = train_one_fold(Xtr, ytr, Xte, yte, in_dim=Xtr.shape[1], class_weights=cw)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(yte, preds)
        f1m = f1_score(yte, preds, average="macro")
        cm = confusion_matrix(yte, preds, labels=list(range(10)))

        fold_csv = out_dir / f"metrics_year_{heldout}.csv"
        pd.DataFrame({"year":[heldout],"accuracy":[acc],"f1_macro":[f1m]}).to_csv(fold_csv, index=False)
        (out_dir / f"classification_report_{heldout}.txt").write_text(classification_report(yte, preds, digits=3))

        rows.append({"year":heldout, "accuracy":acc, "f1_macro":f1m})

    summary = pd.DataFrame(rows).sort_values("year")
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    print(summary)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/lims_prepared.parquet")
    ap.add_argument("--no-state", action="store_true", help="remove State feature columns")
    ap.add_argument("--no-geo", action="store_true", help="remove geography proxy columns")
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    out_dir = Path(args.outdir)
    cfg_tag = ("noState_" if args.no_state else "withState_") + ("noGeo" if args.no_geo else "withGeo")
    out = out_dir / f"loyo_{cfg_tag}"
    out.mkdir(parents=True, exist_ok=True)
    loyo(df, out, use_state=(not args.no_state), use_geo=(not args.no_geo))

if __name__ == "__main__":
    main()
