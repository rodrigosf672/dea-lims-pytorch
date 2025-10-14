import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/lims_prepared.parquet")

# Convert Purity column to numeric, extracting the midpoint of ranges
def convert_purity_to_numeric(purity_str):
    if pd.isna(purity_str):
        return None
    # Handle percentage ranges like "90%-100%"
    if isinstance(purity_str, str) and '%' in purity_str:
        # Remove % signs and split by '-'
        clean_str = purity_str.replace('%', '')
        if '-' in clean_str:
            parts = clean_str.split('-')
            # Take the midpoint of the range
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return float(clean_str)
    return float(purity_str)

df['Purity_numeric'] = df['Purity'].apply(convert_purity_to_numeric)

tier_order = ["Tier1", "Tier2", "Tier3"]
df_tier = df.groupby(["Year","border_tier"])["Purity_numeric"].mean().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=df_tier, x="Year", y="Purity_numeric", hue="border_tier", hue_order=tier_order, marker="o")
plt.title("Average Purity Over Time by Border Tier (2010–2022)", fontsize=16)
plt.ylabel("Average Purity (%)")
plt.legend(title="Border Tier")
plt.tight_layout()
plt.savefig("artifacts/purity_by_tier_over_time.png", dpi=300)
plt.close()