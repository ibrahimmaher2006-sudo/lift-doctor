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


def diagnose(filepath, confidence=0.50, min_consecutive=3):
    """Diagnose like an operator: report the FAULT that developed, and when.
    A fault must persist for several windows IN A ROW (real faults are
    sustained; scattered misclassifications are noise). Biased toward
    catching real faults while rejecting isolated false positives."""
    results = predict_file(filepath, confidence=confidence)

    # Walk the timeline; find the first run of >= min_consecutive same-fault windows
    best_fault, best_onset, best_len = None, None, 0
    run_name, run_start, run_len = None, None, 0

    for start, name, conf in results:
        if name == run_name and name not in ("Normal", "Uncertain"):
            run_len += 1
        else:
            run_name, run_start, run_len = name, start, 1
        # Track the longest qualifying fault run
        if (run_name not in ("Normal", "Uncertain")
                and run_len >= min_consecutive
                and run_len > best_len):
            best_fault, best_onset, best_len = run_name, run_start, run_len

    if best_fault is None:
        return "Normal — no significant fault detected", results, (None, None)

    # Total windows of this fault (for context), and onset time
    total = sum(1 for _, n, _ in results if n == best_fault)
    onset_hr = best_onset / 3600
    msg = (f"{best_fault} — onset at t={best_onset}s (hour {onset_hr:.1f}), "
           f"sustained {best_len}+ consecutive windows ({total} total)")
    return msg, results, (best_fault, best_onset)


if __name__ == "__main__":
    import sys
    from liftdoctor.explain import explain
    f = sys.argv[1] if len(sys.argv) > 1 else None
    if not f:
        print("Usage: python -m liftdoctor.predict <path-to-well-file.parquet>")
    else:
        verdict, detail, (fault_name, onset) = diagnose(f)
        print(f"\nFile: {Path(f).name}")
        print(f"DIAGNOSIS: {verdict}\n")
        print("=" * 60)
        if fault_name:
            print(explain(f, verdict, onset, fault_name))
        else:
            print(explain(f, verdict, 0, "Normal"))
        print("=" * 60)