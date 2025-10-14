import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

BASE_DIR = Path("artifacts")

# LOYO accuracy & F1 over years
def plot_accuracy_f1():
    df = pd.read_csv(BASE_DIR / "loyo_withState_withGeo" / "metrics_summary.csv")
    plt.figure(figsize=(10,5))
    plt.plot(df["year"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["year"], df["f1_macro"], marker="o", label="Macro F1")
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.title("Model Performance Over Time (LOYO)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "performance_over_time.png", dpi=300)
    plt.close()

# Ablation comparison bar chart
def plot_ablation_comparison():
    configs = {
        "withState_withGeo": "Baseline (State + Geo)",
        "withState_noGeo": "No Geo",
        "noState_withGeo": "No State"
    }
    dfs = []
    for key, label in configs.items():
        path = BASE_DIR / f"loyo_{key}" / "metrics_summary.csv"
        if path.exists():
            d = pd.read_csv(path)
            d["config"] = label
            dfs.append(d)
    if not dfs:
        print("No ablation files found.")
        return
    df = pd.concat(dfs)
    df_mean = df.groupby("config")[["accuracy","f1_macro"]].mean().reset_index()

    plt.figure(figsize=(8,5))
    df_mean.set_index("config").plot(kind="bar", ax=plt.gca(), rot=0)
    plt.title("Ablation Study: Geography and State Contribution")
    plt.ylabel("Mean Score (2010–2022)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ablation_comparison.png", dpi=300)
    plt.close()

# Permutation importance (top 15 features)
def plot_permutation_importance():
    path = BASE_DIR / "permimp" / "permutation_importance.csv"
    if not path.exists():
        print("No permutation importance file found.")
        return
    df = pd.read_csv(path).sort_values("acc_drop", ascending=False).head(15)
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x="acc_drop", y="feature", palette="viridis")
    plt.xlabel("Accuracy Drop")
    plt.ylabel("")
    plt.title("Permutation Importance (Top 15 Features)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "permutation_importance_top15.png", dpi=300)
    plt.close()

# Confusion matrices for selected years
def plot_confusion_matrix_for_years(years=[2010,2016,2022]):
    from sklearn.metrics import confusion_matrix
    import torch
    import numpy as np
    from src.train_loyo import make_features
    from src.model import MLP, MLPConfig

    # load full data again
    df = pd.read_parquet("data/lims_prepared.parquet")
    # train model on all except held-out year
    for y in years:
        train = df[df["Year"] != y]
        test = df[df["Year"] == y]
        Xtr, ytr, meta = make_features(train)
        enc = meta["encoder"]
        num_cols = meta["num_cols"]; geo_cols = meta["geo_cols"]; cat_cols = meta["cat_cols"]

        Xte_cat = enc.transform(test[cat_cols].fillna("NA")) if len(cat_cols) else np.zeros((len(test),0))
        Xte_num = test[num_cols + geo_cols].fillna(0).to_numpy(dtype=float)
        Xte = np.hstack([Xte_num, Xte_cat]).astype(np.float32)
        yte = test["purity_idx"].values.astype(int)

        # retrain small model quickly
        from src.train_loyo import train_one_fold, class_weights_from_labels
        cw = class_weights_from_labels(ytr)
        model = train_one_fold(Xtr, ytr, Xte, yte, in_dim=Xtr.shape[1], class_weights=cw)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte).to(device))
            preds = logits.argmax(dim=1).cpu().numpy()

        cm = confusion_matrix(yte, preds, labels=list(range(10)))
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix – Year {y}")
        plt.tight_layout()
        plt.savefig(BASE_DIR / f"confusion_matrix_{y}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    plot_accuracy_f1()
    plot_ablation_comparison()
    plot_permutation_importance()
    plot_confusion_matrix_for_years()
    print("✅ Visualization images saved to artifacts/")
