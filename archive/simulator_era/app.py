"""Lift Doctor — live well monitoring dashboard."""
import streamlit as st
import pandas as pd
from liftdoctor.simulator import ESPWell
from liftdoctor.detector import TrendDetector

st.set_page_config(page_title="Lift Doctor", layout="wide")
st.title("🩺 Lift Doctor — ESP Health Monitor")

# --- Sidebar controls ---
st.sidebar.header("Well Scenario")
failure = st.sidebar.selectbox(
    "Failure mode", [None, "gas_lock", "pump_wear", "tubing_leak"],
    format_func=lambda x: "Healthy well" if x is None else x.replace("_", " ").title(),
)
failure_start = st.sidebar.slider("Failure start (hour)", 13, 22, 18)
seed = st.sidebar.number_input("Random seed", value=42)

# --- Generate and analyze ---
well = ESPWell(name="WELL-01", seed=int(seed))
df = well.generate(hours=24, failure=failure, failure_start_hr=failure_start)

detector = TrendDetector(train_hours=12)
channels = ["intake_psi", "motor_temp_f", "current_amps", "vibration_g"]
results = {ch: detector.score(df[ch]) for ch in channels}

# --- Status banner ---
any_alarm = any(r["anomaly"].any() for r in results.values())
if any_alarm:
    alarmed = [ch for ch in channels if results[ch]["anomaly"].any()]
    first_alert = min(int(results[ch]["anomaly"].idxmax()) for ch in alarmed)
    st.error(f"🚨 ALERT on {', '.join(alarmed)} — first detected at minute {first_alert} "
             f"(hour {first_alert/60:.1f})")
else:
    st.success("✅ All channels nominal")

# --- Charts ---
cols = st.columns(2)
for i, ch in enumerate(channels):
    r = results[ch]
    with cols[i % 2]:
        st.subheader(ch.replace("_", " ").title())
        plot_df = pd.DataFrame({
            "value": r["value"],
            "alert": r["value"].where(r["anomaly"]),
        })
        plot_df.index = df["minute"] / 60
        st.line_chart(plot_df, color=["#1f77b4", "#ff2b2b"], height=250)