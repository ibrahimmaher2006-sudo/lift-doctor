"""Plain-English explanation layer for Lift Doctor diagnoses.
Inspects the real sensor evidence in the fault window and describes it
like an experienced production engineer would."""
import pandas as pd
import numpy as np
from pathlib import Path

# --- Domain knowledge: what each fault does to the sensors, and advice ---
# Each profile lists the sensors that characterize the fault, the expected
# direction of change, and a human description + recommended action.
FAULT_PROFILES = {
    "BSW increase": {
        "watch": ["P-MON-CKP", "T-TPT", "QGL"],
        "summary": "rising water cut (BSW) in the produced fluid",
        "physics": "an increasing water fraction altering flow density and "
                   "wellhead conditions",
        "action": "Verify water cut via sampling; review the well's water "
                  "management and consider rate or lift adjustments.",
        "confusable": None,
    },
    "DHSV closure": {
        "watch": ["P-PDG", "P-TPT", "P-MON-CKP"],
        "summary": "a spurious closure of the downhole safety valve (DHSV)",
        "physics": "the DHSV restricting flow causes a sharp pressure response "
                   "across the wellbore",
        "action": "Inspect the DHSV control line and actuation; confirm valve "
                  "state before resuming normal operation.",
        "confusable": None,
    },
    "Severe slugging": {
        "watch": ["P-MON-CKP", "QGL", "P-PDG"],
        "summary": "severe slugging — large oscillating swings in flow and pressure",
        "physics": "alternating gas and liquid slugs create cyclic pressure "
                   "and flow surges",
        "action": "Stabilize with choke/gas-lift adjustment; consider slug "
                  "mitigation or topside control changes.",
        "confusable": "flow instability",
    },
    "Flow instability": {
        "watch": ["P-MON-CKP", "QGL"],
        "summary": "unstable multiphase flow in the production line",
        "physics": "irregular fluctuations in pressure and gas-lift flow "
                   "without a clean slugging cycle",
        "action": "Review choke setting and gas-lift rate to restore stable flow.",
        "confusable": "severe slugging",
    },
    "Rapid productivity loss": {
        "watch": ["P-MON-CKP", "T-TPT", "QGL", "T-JUS-CKP"],
        "summary": "a rapid loss of well productivity",
        "physics": "falling choke pressure and cooling temperatures as flow "
                   "declines, often with gas-lift disruption",
        "action": "Investigate inflow problems, gas-lift performance, and "
                  "possible near-wellbore restrictions urgently.",
        "confusable": "hydrate or scaling",
    },
    "Quick PCK restriction": {
        "watch": ["P-MON-CKP", "P-JUS-CKGL", "ABER-CKP"],
        "summary": "a quick restriction in the production choke (PCK)",
        "physics": "a sudden flow restriction at the choke shifts the pressure "
                   "drop across it",
        "action": "Inspect the production choke for obstruction or actuation "
                  "fault.",
        "confusable": "flow instability",
    },
    "PCK scaling": {
        "watch": ["P-MON-CKP", "P-JUS-CKGL", "T-JUS-CKP"],
        "summary": "scale build-up in the production choke (PCK)",
        "physics": "gradual mineral scaling slowly increases the choke "
                   "restriction over time",
        "action": "Plan a choke inspection and descaling; review scale-inhibitor "
                  "program.",
        "confusable": "normal operation (early scaling is subtle)",
    },
    "Hydrate in line": {
        "watch": ["P-MON-CKP", "T-TPT", "T-JUS-CKP", "QGL"],
        "summary": "hydrate formation in the production line",
        "physics": "low temperatures with pressure changes promote hydrate "
                   "plugs that restrict flow",
        "action": "Apply hydrate remediation (heat, depressurization, or "
                  "inhibitor injection) and review thermal management.",
        "confusable": "rapid productivity loss",
    },
    "Event type 9": {
        "watch": ["P-MON-CKP", "P-PDG", "T-TPT"],
        "summary": "an anomaly matching event-type-9 patterns",
        "physics": "a deviation from normal operation flagged by the model",
        "action": "Review recent operational changes and sensor trends to "
                  "characterize the event.",
        "confusable": None,
    },
}


def _sensor_evidence(df, onset_sec, window_sec=300):
    """Compare sensor behavior BEFORE vs AFTER the fault onset.
    Uses a wider 'after' window so gradual faults still show evidence."""
   # 'before' = the well's healthy baseline (first 30 min of the record);
    # 'after' = 30 min from fault onset. For slow faults, this captures the
    # cumulative drift that a narrow before/after window misses.
    before = df.iloc[0:1800]
    after = df.iloc[onset_sec:onset_sec + 1800]
    evidence = {}
    for col in df.columns:
        if col in ("class", "state"):
            continue
        if before[col].notna().sum() > 5 and after[col].notna().sum() > 5:
            b = before[col].mean()
            a = after[col].mean()
            if b != 0 and not np.isnan(b) and not np.isnan(a):
                evidence[col] = (a - b) / abs(b) * 100
    return evidence


def explain(filepath, diagnosis_str, onset_sec, fault_name, confidence_note=""):
    """Build a plain-English explanation of a diagnosis using real evidence."""
    df = pd.read_parquet(filepath)
    well = Path(filepath).stem

    if fault_name == "Normal" or fault_name not in FAULT_PROFILES:
        return (f"{well}: No significant fault detected. The well's sensors "
                f"remained within normal operating patterns throughout the record.")

    profile = FAULT_PROFILES[fault_name]
    evidence = _sensor_evidence(df, onset_sec)
    onset_hr = onset_sec / 3600

    # Describe the most-moved watched sensors
    moves = []
    for s in profile["watch"]:
        if s in evidence and abs(evidence[s]) >= 5:   # only notable moves
            direction = "rose" if evidence[s] > 0 else "fell"
            moves.append(f"{s} {direction} {abs(evidence[s]):.0f}%")

    # Assemble the explanation
    lines = []
    lines.append(f"**{well} — {fault_name}**")
    lines.append(f"Detected at hour {onset_hr:.1f}. The model identifies "
                 f"{profile['summary']}.")
    if moves:
        lines.append(f"Evidence around onset: {', '.join(moves)} — consistent "
                     f"with {profile['physics']}.")
    else:
        lines.append(f"This is consistent with {profile['physics']}, though the "
                     f"signature was subtle at onset.")
    lines.append(f"Recommended action: {profile['action']}")
    if profile["confusable"]:
        lines.append(f"Note: this pattern can resemble {profile['confusable']}; "
                     f"confirm before acting.")
    if confidence_note:
        lines.append(confidence_note)
    return "\n".join(lines)