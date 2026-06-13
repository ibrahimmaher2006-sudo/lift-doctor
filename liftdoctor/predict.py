"""Predict the failure type for a raw 3W well file, using the saved pipeline."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from liftdoctor.threew import SENSORS, summarize_window

_BUNDLE = None

def _load():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load("liftdoctor/pipeline_3w.pkl")
    return _BUNDLE


def _prep_one_window(window, bundle):
    """Turn one window of raw sensor data into a model-ready feature row.
    Keeps column names intact all the way through (critical for correctness)."""
    feats = summarize_window(window)
    cols = bundle["feature_cols"]
    # Build a one-row frame in EXACTLY the training column order
    row = pd.DataFrame([feats]).reindex(columns=cols)
    row = row.replace([np.inf, -np.inf], np.nan)
    for col in cols:
        lo, hi = bundle["clip_bounds"][col]
        row[col] = row[col].clip(lo, hi)
    row = row.fillna(-999)
    # Scale, then REBUILD a named DataFrame so the model sees feature names
    scaled = bundle["scaler"].transform(row)
    return pd.DataFrame(scaled, columns=cols)


def predict_file(filepath, window_sec=300, confidence=0.50):
    """Diagnose a well file window-by-window.
    Returns a list of (window_start_sec, diagnosis, confidence)."""
    bundle = _load()
    df = pd.read_parquet(filepath)
    results = []
    for start in range(0, len(df) - window_sec, window_sec):
        window = df.iloc[start:start + window_sec]
        X = _prep_one_window(window, bundle)
        probs = bundle["model"].predict_proba(X)[0]
        top = probs.argmax()
        event = bundle["model"].classes_[top]
        conf = float(probs[top])
        name = bundle["event_names"][event]
        if conf < confidence:
            name = "Uncertain"
        results.append((start, name, round(conf, 3)))
    return results


def diagnose(filepath, confidence=0.50, min_fault_windows=3):
    """Diagnose a well like an operator: report the FAULT that developed,
    not the most common state. A few confident fault windows = a finding."""
    results = predict_file(filepath, confidence=confidence)
    confident = [(s, name, c) for s, name, c in results if name != "Uncertain"]
    if not confident:
        return "Uncertain — no confident diagnosis", results

    # Separate normal windows from actual faults
    fault_windows = [(s, name, c) for s, name, c in confident if name != "Normal"]

    # Count each fault type; require a minimum to avoid one-off noise
    from collections import Counter
    fault_counts = Counter(name for _, name, _ in fault_windows)
    real_faults = {f: n for f, n in fault_counts.items() if n >= min_fault_windows}

    if not real_faults:
        return "Normal — no significant fault detected", results

    # The dominant fault is the diagnosis
    top_fault = max(real_faults, key=real_faults.get)

    # WHEN did it first appear? (early detection — the valuable part)
    onset = next(s for s, name, c in fault_windows if name == top_fault)
    n_fault = real_faults[top_fault]
    onset_hr = onset / 3600

    return (f"{top_fault} — detected at t={onset}s (hour {onset_hr:.1f}), "
            f"sustained over {n_fault} windows"), results


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else None
    if not f:
        print("Usage: python -m liftdoctor.predict <path-to-well-file.parquet>")
    else:
        verdict, detail = diagnose(f)
        print(f"\nFile: {Path(f).name}")
        print(f"DIAGNOSIS: {verdict}\n")
        print("Window-by-window:")
        for start, name, conf in detail[:20]:
            print(f"  t={start:5d}s : {name:25s} (conf {conf})")