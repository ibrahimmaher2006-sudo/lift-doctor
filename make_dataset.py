"""Generates a labeled training dataset from simulated wells."""
import numpy as np
import pandas as pd
from liftdoctor.simulator import ESPWell

CHANNELS = ["intake_psi", "motor_temp_f", "current_amps", "vibration_g"]
FAILURES = [None, "gas_lock", "pump_wear", "tubing_leak"]
TRAIN_MIN = 12 * 60   # baseline = first 12 hours


def extract_features(df, start_min):
    """Boil a well history down to one row of numbers,
    looking only at the first hour after failure onset."""
    feats = {}
    window = {}
    for ch in CHANNELS:
        base = df[ch].iloc[:TRAIN_MIN]
        late = df[ch].iloc[start_min:start_min + 60]   # first hour after onset
        mu, sigma = base.mean(), base.std()
        feats[f"{ch}_shift"] = (late.mean() - mu) / sigma     # how far it moved
        feats[f"{ch}_slope"] = np.polyfit(range(len(late)), late, 1)[0]  # trend
        feats[f"{ch}_noise"] = late.std() / sigma             # got more erratic?
        window[ch] = late.reset_index(drop=True)

    # --- Cross-channel features: relationships, not just individual moves ---
    p, c = window["intake_psi"], window["current_amps"]
    v, temp = window["vibration_g"], window["motor_temp_f"]

    # 1. Pressure vs current correlation: gas lock = both erratic/opposite
    feats["psi_current_corr"] = p.corr(c)

    # 2. Direction of pressure move (up=leak, down=gas lock) signed by current
    psi_dir = np.sign(feats["intake_psi_shift"])
    cur_dir = np.sign(feats["current_amps_shift"])
    feats["psi_current_agree"] = psi_dir * cur_dir   # +1 same dir, -1 opposite

    # 3. Vibration vs temperature correlation: pump wear = both rise together
    feats["vib_temp_corr"] = v.corr(temp)

    # 4. Mechanical vs hydraulic dominance: is the action in vib/temp or psi/current?
    mech = abs(feats["vibration_g_shift"]) + abs(feats["motor_temp_f_shift"])
    hyd = abs(feats["intake_psi_shift"]) + abs(feats["current_amps_shift"])
    feats["mech_vs_hyd"] = mech - hyd

    return feats


def main(n_wells=600, out="data/training.csv"):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_wells):
        failure = FAILURES[i % 4]                      # balanced classes
        start_hr = int(rng.integers(14, 21))
        well = ESPWell(seed=int(rng.integers(0, 1_000_000)))
        df = well.generate(hours=24, failure=failure, failure_start_hr=start_hr)
        feats = extract_features(df, start_hr * 60)
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