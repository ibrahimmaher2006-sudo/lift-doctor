"""Lift Doctor 3W — failure diagnosis dashboard for real offshore well data."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from liftdoctor.predict import predict_file, diagnose
from liftdoctor.explain import explain, FAULT_PROFILES

DATA_DIR = Path("/Users/ibrahimshehab/Desktop/3W-main/dataset")
EVENT_NAMES = {
    0: "Normal", 1: "BSW increase", 2: "DHSV closure", 3: "Severe slugging",
    4: "Flow instability", 5: "Rapid productivity loss", 6: "Quick PCK restriction",
    7: "PCK scaling", 8: "Hydrate in line", 9: "Event type 9",
}

st.set_page_config(page_title="Lift Doctor 3W", layout="wide")
st.title("🛢️ Lift Doctor — Offshore Well Failure Diagnosis")
st.caption("Real-time fault detection on Petrobras 3W field data")

# --- Sidebar: pick a well ---
st.sidebar.header("Select a Well")
event = st.sidebar.selectbox(
    "Event folder", list(range(10)),
    format_func=lambda i: f"{i} — {EVENT_NAMES[i]}",
)
folder = DATA_DIR / str(event)
files = sorted([f.name for f in folder.glob("WELL-*.parquet")]) or \
        sorted([f.name for f in folder.glob("*.parquet")])

if not files:
    st.error(f"No files found in folder {event}.")
    st.stop()

filename = st.sidebar.selectbox("Well file", files)
filepath = folder / filename
st.sidebar.write(f"Ground truth: **{EVENT_NAMES[event]}**")

confidence = st.sidebar.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05)

# Cache the EXPENSIVE part (running the model on every window) per file only.
@st.cache_data(show_spinner="Analyzing well data (first time for this well)...")
def get_raw_predictions(path_str):
    # confidence=0 so nothing is marked Uncertain here; we threshold later
    return predict_file(path_str, confidence=0.0)

raw = get_raw_predictions(str(filepath))

# Apply the confidence threshold cheaply (instant — no model calls)
results = [(s, (n if c >= confidence else "Uncertain"), c) for s, n, c in raw]

# Find the longest CONSECUTIVE run of the same fault (real persistence logic,
# matching diagnose() — rejects scattered noise on healthy wells)
MIN_CONSEC = 3
best_fault, best_onset, best_len = None, None, 0
run_name, run_start, run_len = None, None, 0
for s, name, c in results:
    if name == run_name and name not in ("Normal", "Uncertain"):
        run_len += 1
    else:
        run_name, run_start, run_len = name, s, 1
    if (run_name not in ("Normal", "Uncertain")
            and run_len >= MIN_CONSEC and run_len > best_len):
        best_fault, best_onset, best_len = run_name, run_start, run_len

if best_fault:
    fault_name, onset = best_fault, best_onset
    total = sum(1 for _, n, _ in results if n == best_fault)
    onset_hr = onset / 3600
    verdict = (f"{best_fault} — onset at t={onset}s (hour {onset_hr:.1f}), "
               f"sustained {best_len}+ consecutive windows ({total} total)")
else:
    fault_name, onset = None, None
    verdict = "Normal — no significant fault detected"

# Re-derive the verdict from thresholded results (fast)
from collections import Counter
fault_runs = [(s, n) for s, n, c in results if n not in ("Normal", "Uncertain")]
if fault_runs:
    fault_name = Counter(n for _, n in fault_runs).most_common(1)[0][0]
    onset = next(s for s, n in fault_runs if n == fault_name)
    onset_hr = onset / 3600
    n_fault = sum(1 for _, n in fault_runs if n == fault_name)
    verdict = (f"{fault_name} — onset at t={onset}s (hour {onset_hr:.1f}), "
               f"{n_fault} fault windows")
else:
    fault_name, onset = None, None
    verdict = "Normal — no significant fault detected"

# --- Diagnosis banner ---
if fault_name:
    st.error(f"🚨 **DIAGNOSIS: {verdict}**")
else:
    st.success(f"✅ **{verdict}**")

# --- Two columns: explanation + prediction timeline ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Engineer's Assessment")
    if fault_name:
        text = explain(str(filepath), verdict, onset, fault_name)
    else:
        text = explain(str(filepath), verdict, 0, "Normal")
    st.markdown(text)

with col2:
    st.subheader("🔍 Prediction Timeline")
    # Build a per-window dataframe of predictions
    timeline = pd.DataFrame(results, columns=["t_sec", "prediction", "confidence"])
    timeline["hour"] = timeline["t_sec"] / 3600
    # Count predictions
    counts = Counter(timeline["prediction"])
    st.write("**Windows by prediction:**")
    for label, n in counts.most_common():
        st.write(f"- {label}: {n} windows")

# --- Sensor charts with fault onset marked ---
st.subheader("📈 Sensor Data")
df = pd.read_parquet(filepath)

# Pick the most informative sensors that have data
KEY_SENSORS = ["P-MON-CKP", "P-PDG", "T-TPT", "T-JUS-CKP", "QGL", "P-TPT"]
available = [s for s in KEY_SENSORS if s in df.columns and df[s].notna().any()]

# Normalize each sensor 0-1 for display on one chart (they're wildly different scales)
# Downsample to ~1000 points for fast, smooth charts
# Downsample to ~1000 points for fast, smooth charts
step = max(1, len(df) // 1000)
idx = np.arange(0, len(df), step)
chart_df = pd.DataFrame()
chart_df["hour"] = idx / 3600
for s in available[:4]:
    vals = df[s].ffill().bfill().values.astype(float)[idx]
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    rng = vmax - vmin
    if rng > 1e-6 and np.isfinite(rng):
        chart_df[s] = (vals - vmin) / rng        # normal case: scale 0–1
    else:
        chart_df[s] = 0.5                          # flat sensor: show a flat mid-line
chart_df = chart_df.set_index("hour")
st.line_chart(chart_df, height=300)
st.caption(f"Sensors shown (normalized 0–1): {', '.join(available[:4])}. "
           + (f"Fault onset detected at hour {onset/3600:.1f}." if fault_name
              else "No fault detected."))

# --- Confidence-over-time chart ---
st.subheader("📊 Model Confidence Over Time")
conf_df = timeline[["hour", "confidence"]].set_index("hour")
st.area_chart(conf_df, height=200)
st.caption("Higher = model is more certain of its window prediction. "
           "Dips indicate ambiguous periods (often the transition into a fault).")

# --- Footer ---
st.divider()
st.caption("Lift Doctor 3W · Gradient-boosted classifier · ~85% accuracy on "
           "unseen wells · Trained on Petrobras 3W dataset")