import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("data/lims_prepared.parquet")

# Helper function for midpoint conversion
def convert_purity_to_numeric(purity_str):
    if pd.isna(purity_str):
        return None
    if isinstance(purity_str, str):
        s = purity_str.strip().replace('%', '').strip()
        if s == '':
            return None
        if '-' in s:
            parts = s.split('-')
            try:
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(purity_str)
    except:
        return None

# Clean and aggregate
df['Purity_numeric'] = df['Purity'].apply(convert_purity_to_numeric)
df['State'] = df['State'].astype(str).str.strip()

# Compute mean purity by state
purity_by_state = (
    df[df['Purity_numeric'].notna()]
    .groupby('State', as_index=False)['Purity_numeric']
    .mean()
)

# Sort states by purity for easier visualization
purity_by_state = purity_by_state.sort_values('Purity_numeric', ascending=False)

# Plot heatmap
plt.figure(figsize=(10, 16))
sns.heatmap(
    purity_by_state.set_index('State'),
    cmap="YlOrRd",
    annot=True,
    fmt=".1f",
    cbar_kws={'label': 'Mean Purity (%)'}
)
plt.title('Mean Drug Purity by State (2010–2022)')
plt.ylabel('')
plt.xlabel('')
plt.tight_layout()
plt.savefig('artifacts/heatmap_purity_by_state.png', dpi=300)
plt.close()
