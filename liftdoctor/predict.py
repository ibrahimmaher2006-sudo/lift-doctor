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
    """Turn one window of raw sensor data into a model-ready feature row."""
    feats = summarize_window(window)                 # same features as training
    row = pd.DataFrame([feats]).reindex(columns=bundle["feature_cols"])
    # apply the SAME clipping + scaling learned at training time
    row = row.replace([np.inf, -np.inf], np.nan)
    for col in bundle["feature_cols"]:
        lo, hi = bundle["clip_bounds"][col]
        row[col] = row[col].clip(lo, hi)
    row = row.fillna(-999)
    return bundle["scaler"].transform(row).astype("float32")


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
        conf = probs[top]
        name = bundle["event_names"][event]
        if conf < confidence:
            name = "Uncertain"
        results.append((start, name, round(float(conf), 3)))
    return results


def diagnose(filepath, confidence=0.50):
    """Give one overall diagnosis for a well file (most common confident call)."""
    results = predict_file(filepath, confidence=confidence)
    confident = [r[1] for r in results if r[1] != "Uncertain"]
    if not confident:
        return "Uncertain — no confident diagnosis", results
    verdict = pd.Series(confident).mode().iloc[0]
    share = confident.count(verdict) / len(confident)
    return f"{verdict} (in {share:.0%} of confident windows)", results


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