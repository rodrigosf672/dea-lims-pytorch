
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BORDER_STATES = {"AZ","CA","NM","TX"}
NEIGHBOR_OF_BORDER = {
    "NV","UT","CO","OK","LA","AR"  # neighbors touching AZ, CA, NM, TX
}

PURITY_ORDER = [
    "0%-10%","10%-20%","20%-30%","30%-40%","40%-50%",
    "50%-60%","60%-70%","70%-80%","80%-90%","90%-100%"
]

def normalize_purity_bin(x: str):
    if pd.isna(x) or x == "":
        return np.nan
    x = str(x).strip()
    if re.match(r"^\d+%-\d+%$", x):
        return x
    # normalize variants like "0 to 10%"
    x = x.replace("to","-").replace(" ","")
    if x.endswith("%") and "-" in x:
        return x
    nums = re.findall(r"\d{1,3}", x)
    if len(nums) == 2:
        return f"{int(nums[0])}% - {int(nums[1])}%".replace(" ","").replace("%-","%-")
    return np.nan

def month_to_sin_cos(m):
    if pd.isna(m):
        return (np.nan, np.nan)
    k = (int(m) - 1) / 12 * 2*np.pi
    return np.sin(k), np.cos(k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to LIMS XLS or CSV")
    ap.add_argument("--out", default="data/lims_prepared.parquet")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    if in_path.suffix.lower() in [".xls",".xlsx"]:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Canonical column names
    rename = {
        "drug_type":"Drug_Type","DrugType":"Drug_Type","Drug_Type":"Drug_Type",
        "net_weight":"Net_Weight","Net_Weight":"Net_Weight",
        "state":"State","State":"State",
        "price":"Price","Price":"Price",
        "year":"Year","Year":"Year",
        "month":"Month","Month":"Month",
        "purity":"Purity","Purity":"Purity"
    }
    for k,v in list(rename.items()):
        if k in df.columns and k != v:
            df.rename(columns={k:v}, inplace=True)

    df["Purity"] = df["Purity"].astype(str).replace({"nan":"", "None":""})
    df["Purity"] = df["Purity"].apply(normalize_purity_bin)
    df = df[~df["Purity"].isna()].copy()
    df = df[df["Purity"].isin(PURITY_ORDER)].copy()
    df["purity_idx"] = df["Purity"].map({b:i for i,b in enumerate(PURITY_ORDER)})

    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Year"]  = pd.to_numeric(df["Year"], errors="coerce")

    df["State"] = df["State"].astype(str).str.upper().str.replace(r"[^A-Z/]", "", regex=True)
    df["State"] = df["State"].str.replace("ND/NE/SD/WY","ND_NE_SD_WY")

    df["is_border_state"] = df["State"].isin(BORDER_STATES).astype(int)
    df["is_neighbor_of_border"] = df["State"].isin(NEIGHBOR_OF_BORDER).astype(int)
    df["border_tier"] = np.where(df["is_border_state"]==1, 0,
                          np.where(df["is_neighbor_of_border"]==1, 1, 2))

    s, c = zip(*df["Month"].map(month_to_sin_cos))
    df["month_sin"] = s
    df["month_cos"] = c

    for col in ["Drug_Type","Net_Weight","Price"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    cols_keep = [
        "Year","Month","Drug_Type","Net_Weight","Price","State",
        "is_border_state","is_neighbor_of_border","border_tier",
        "month_sin","month_cos","Purity","purity_idx"
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep].dropna(subset=["Year"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(df):,} rows.")

if __name__ == "__main__":
    main()
