import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from .train_loyo import make_features, class_weights_from_labels, train_one_fold

def permutation_importance(df: pd.DataFrame, outdir: Path):
    base_year = 2022 if 2022 in df["Year"].unique() else int(df["Year"].max())
    tr = df[df["Year"] != base_year].reset_index(drop=True)
    te = df[df["Year"] == base_year].reset_index(drop=True)

    Xtr, ytr, meta = make_features(tr)
    enc = meta["encoder"]; num_cols = meta["num_cols"]; geo_cols = meta["geo_cols"]; cat_cols = meta["cat_cols"]
    Xte_cat = enc.transform(te[cat_cols].fillna("NA")) if len(cat_cols) else np.zeros((len(te),0))
    Xte_num = te[num_cols + geo_cols].fillna(0).to_numpy(dtype=float)
    Xte = np.hstack([Xte_num, Xte_cat]).astype(np.float32); yte = te["purity_idx"].values.astype(int)

    cw = class_weights_from_labels(ytr, 10)
    model = train_one_fold(Xtr, ytr, Xte, yte, in_dim=Xtr.shape[1], class_weights=cw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        base = model(torch.from_numpy(Xte).to(device)).argmax(1).cpu().numpy()
    base_acc = accuracy_score(yte, base)

    names = meta["feature_names"]
    drops = []
    rng = np.random.default_rng(123)
    for j, name in enumerate(names):
        Xpert = Xte.copy()
        rng.shuffle(Xpert[:, j])
        with torch.no_grad():
            preds = model(torch.from_numpy(Xpert).to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(yte, preds)
        drops.append({"feature": name, "acc_drop": base_acc - acc})

    out = pd.DataFrame(drops).sort_values("acc_drop", ascending=False)
    out.to_csv(outdir / "permutation_importance.csv", index=False)
    print(out.head(20))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="artifacts/permimp")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    permutation_importance(df, outdir)

if __name__ == "__main__":
    main()
