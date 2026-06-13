"""Inspect how sensors behave across normal/transient/fault stages."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset")
f = DATA_DIR / "5" / "WELL-00015_20170620122925.parquet"

df = pd.read_parquet(f)

# Key pressure/temp/flow sensors to watch
KEY = ["P-PDG", "P-TPT", "P-MON-CKP", "T-TPT", "QGL"]
key = [c for c in KEY if c in df.columns and df[c].notna().any()]

# Label each stage
stage_names = {0: "NORMAL", 105: "TRANSIENT (failing)", 5: "FAULT (failed)"}

print(f"File: {f.name}  ({len(df)} seconds)\n")
print("How long the well spends in each stage:")
print(df["class"].value_counts().sort_index().to_string())
print()

# For each stage, show the average value of each key sensor
print("Average sensor values per stage:")
print("-" * 70)
summary = df.groupby("class")[key].mean()
summary.index = [stage_names.get(i, i) for i in summary.index]
print(summary.to_string())
print()

# The key question: how much does each sensor MOVE from normal to fault?
if 0 in df["class"].values and 5 in df["class"].values:
    normal = df[df["class"] == 0][key].mean()
    fault = df[df["class"] == 5][key].mean()
    pct_change = ((fault - normal) / normal * 100).round(1)
    print("% change in each sensor from NORMAL to FAULT:")
    print("-" * 70)
    for sensor in key:
        print(f"  {sensor:12s}: {pct_change[sensor]:+7.1f}%")