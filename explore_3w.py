"""First look at the 3W dataset — structure and one sample file."""
import pandas as pd
from pathlib import Path

# --- Point this at wherever your 3W data folders live ---
# Common spots: a "dataset" folder, or numbered folders 0,1,2...
DATA_DIR = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset")

# 1. What's in the data directory?
print("=== Folder structure ===")
for item in sorted(DATA_DIR.rglob("*")):
    if item.is_dir():
        n_files = len(list(item.glob("*.parquet"))) + len(list(item.glob("*.csv")))
        print(f"  {item.relative_to(DATA_DIR)}/   ({n_files} data files)")

# 2. Find the first data file anywhere under DATA_DIR
all_files = list(DATA_DIR.rglob("*.parquet")) + list(DATA_DIR.rglob("*.csv"))
print(f"\n=== Total data files found: {len(all_files)} ===")

if not all_files:
    print("No .parquet or .csv files found — check DATA_DIR path.")
else:
    sample = all_files[0]
    print(f"\n=== Previewing: {sample} ===")
    if sample.suffix == ".parquet":
        df = pd.read_parquet(sample)
    else:
        df = pd.read_csv(sample)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nColumn data types:\n{df.dtypes}")
    print(f"\nMissing values per column:\n{df.isna().sum()}")