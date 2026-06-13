"""Loader for the Petrobras 3W dataset.
Turns raw multi-sensor well files into labeled feature rows for ML.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset")

# The sensor channels we care about (the continuous measurements).
# We deliberately skip ESTADO-* (equipment on/off states) for now.
SENSORS = [
    "P-PDG", "P-TPT", "P-MON-CKP", "P-JUS-CKGL", "P-ANULAR",
    "T-PDG", "T-TPT", "T-JUS-CKP", "QGL", "ABER-CKP", "ABER-CKGL",
]

EVENT_NAMES = {
    0: "Normal", 1: "BSW increase", 2: "DHSV closure", 3: "Severe slugging",
    4: "Flow instability", 5: "Rapid productivity loss", 6: "Quick PCK restriction",
    7: "PCK scaling", 8: "Hydrate in line", 9: "Event type 9",
}


def stage_of(class_value, event_folder):
    """Map a raw per-second class code to a stage label.
    Codes: 0 = normal. N = established fault N. 100+N = transient into fault N.
    Returns a string like 'normal', 'transient_5', or 'fault_5'.
    """
    if pd.isna(class_value):
        return None
    c = int(class_value)
    if c == 0:
        return "normal"
    elif c >= 100:
        return f"transient_{c - 100}"
    else:
        return f"fault_{c}"


def summarize_window(window):
    """Turn a chunk of multi-sensor readings into one row of features.
    For each sensor that has data: mean, std (noisiness), and slope (trend).
    Missing sensors get NaN, which we handle later.
    """
    feats = {}
    x = np.arange(len(window))
    for s in SENSORS:
        if s in window.columns and window[s].notna().sum() > 5:
            vals = window[s].astype(float).ffill().bfill().values
            feats[f"{s}_mean"] = np.mean(vals)
            feats[f"{s}_std"] = np.std(vals)
            # slope via least-squares fit (trend over the window)
            feats[f"{s}_slope"] = np.polyfit(x, vals, 1)[0]
        else:
            feats[f"{s}_mean"] = np.nan
            feats[f"{s}_std"] = np.nan
            feats[f"{s}_slope"] = np.nan
    return feats


def load_file_windows(filepath, event_folder, window_sec=300, step_sec=300):
    """Slice one well file into fixed-length windows, summarize + label each.
    window_sec=300 -> each row summarizes 5 minutes of well behavior.
    """
    df = pd.read_parquet(filepath)
    if "class" not in df.columns:
        return []

    rows = []
    for start in range(0, len(df) - window_sec, step_sec):
        window = df.iloc[start:start + window_sec]
        # Label the window by its MOST COMMON class (the dominant stage)
        stages = window["class"].apply(lambda c: stage_of(c, event_folder))
        stages = stages.dropna()
        if len(stages) == 0:
            continue
        label = stages.mode().iloc[0]

        feats = summarize_window(window)
        feats["label"] = label
        feats["source_file"] = filepath.name
        rows.append(feats)
    return rows


def build_dataset(files_per_event=15, window_sec=300, seed=0):
    """Sample files across all event folders and build one labeled table."""
    rng = np.random.default_rng(seed)
    all_rows = []

    for event in range(10):
        folder = DATA_DIR / str(event)
        if not folder.exists():
            continue
        files = sorted(folder.glob("*.parquet"))
        if not files:
            continue
        # Sample a manageable number of files from this event type
        n = min(files_per_event, len(files))
        chosen = rng.choice(files, size=n, replace=False)
        print(f"Event {event} ({EVENT_NAMES[event]}): {n} of {len(files)} files")

        for f in chosen:
            all_rows.extend(load_file_windows(Path(f), event, window_sec))

    dataset = pd.DataFrame(all_rows)
    print(f"\nBuilt {len(dataset)} windows total.")
    print(f"\nLabel distribution:\n{dataset['label'].value_counts().to_string()}")
    return dataset


if __name__ == "__main__":
    data = build_dataset(files_per_event=60, window_sec=300, seed=0)
    out = "data/threew_features.csv"
    data.to_csv(out, index=False)
    print(f"\nSaved to {out}")