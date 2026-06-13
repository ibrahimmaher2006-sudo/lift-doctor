"""Generates a labeled training dataset from simulated wells."""
import numpy as np
import pandas as pd
from liftdoctor.simulator import ESPWell

CHANNELS = ["intake_psi", "motor_temp_f", "current_amps", "vibration_g"]
FAILURES = [None, "gas_lock", "pump_wear", "tubing_leak"]
TRAIN_MIN = 12 * 60   # baseline = first 12 hours


def extract_features(df):
    """Boil a 24h well history down to one row of numbers."""
    feats = {}
    for ch in CHANNELS:
        base = df[ch].iloc[:TRAIN_MIN]
        late = df[ch].iloc[-120:]                      # last 2 hours
        mu, sigma = base.mean(), base.std()
        feats[f"{ch}_shift"] = (late.mean() - mu) / sigma     # how far it moved
        feats[f"{ch}_slope"] = np.polyfit(range(len(late)), late, 1)[0]  # trend
        feats[f"{ch}_noise"] = late.std() / sigma             # got more erratic?
    return feats


def main(n_wells=600, out="data/training.csv"):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_wells):
        failure = FAILURES[i % 4]                      # balanced classes
        start_hr = int(rng.integers(14, 21))
        well = ESPWell(seed=int(rng.integers(0, 1_000_000)))
        df = well.generate(hours=24, failure=failure, failure_start_hr=start_hr)
        feats = extract_features(df)
        feats["label"] = failure or "healthy"
        rows.append(feats)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n_wells} wells simulated...")

    dataset = pd.DataFrame(rows)
    dataset.to_csv(out, index=False)
    print(f"\nSaved {len(dataset)} labeled wells to {out}")
    print(dataset["label"].value_counts())


if __name__ == "__main__":
    main()