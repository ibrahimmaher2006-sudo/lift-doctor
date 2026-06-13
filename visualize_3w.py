"""Visualize real 3W fault events — watch a well fail over time."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset")

# Event type names (folder number -> meaning)
EVENT_NAMES = {
    0: "Normal", 1: "BSW increase", 2: "DHSV closure", 3: "Severe slugging",
    4: "Flow instability", 5: "Rapid productivity loss", 6: "Quick PCK restriction",
    7: "PCK scaling", 8: "Hydrate in line", 9: "Event type 9",
}

# Pick one folder to visualize. Start with 5 = rapid productivity loss.
EVENT = 5

# Prefer a REAL well file (starts with WELL-), not a SIMULATED one
folder = DATA_DIR / str(EVENT)
real_files = sorted(folder.glob("WELL-*.parquet"))
files_to_use = real_files[:1] if real_files else sorted(folder.glob("*.parquet"))[:1]

for f in files_to_use:
    df = pd.read_parquet(f)
    print(f"\nFile: {f.name}")
    print(f"Rows: {len(df)}, Duration: ~{len(df)/3600:.1f} hours (at 1 Hz)")
    print(f"Class values present: {sorted(df['class'].dropna().unique())}")

    # Keep only sensor columns that actually have data in this file
    sensor_cols = [c for c in df.columns
                   if c not in ("class", "state") and df[c].notna().any()]
    print(f"Sensors with data: {sensor_cols}")

    # Plot each working sensor, with the fault period shaded
    n = len(sensor_cols)
    fig, axes = plt.subplots(n + 1, 1, figsize=(13, 2.2 * (n + 1)), sharex=True)

    x = range(len(df))
    for ax, col in zip(axes[:-1], sensor_cols):
        ax.plot(x, df[col].values, linewidth=0.6)