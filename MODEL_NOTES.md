# Lift Doctor — 3W Model Notes

## Model
- Task: 10-class failure classification on Petrobras 3W offshore well data
- Algorithm: Random Forest (200 trees, class_weight="balanced")
- Features: per-sensor mean, std, slope over 5-min windows (11 sensors)
- Validation: grouped split by well file (no data leakage)

## Honest Performance
- **Overall accuracy: ~83%** on wells never seen in training
- Strong classes (F1 > 0.90): BSW increase, DHSV closure, severe slugging,
  quick PCK restriction, hydrate in line, event type 9
- Moderate: rapid productivity loss (0.87), flow instability (0.85)
- Weak: PCK scaling (0.80), Normal precision (0.49)

## Known Limitations
1. **Established PCK scaling is not reliably detected** (recall ~0%).
   - Root cause: severe sub-class imbalance. The dataset contains ~28,900
     early-scaling (transient) windows but only ~440 established-scaling
     windows. The model cannot learn the established phase from so few examples.
   - Impact: LOW. The product targets EARLY detection; early scaling is
     detected at F1 0.81. Established scaling is the post-failure state an
     operator would already be aware of.
   - Decision: accepted as a data limitation rather than fixed artificially,
     to keep the model honest and avoid overfitting to ~440 windows.

2. **Normal vs early-slow-failure overlap.** Early-stage slow failures (scaling,
   some hydrate) resemble normal operation by nature; some confusion is inherent
   to the physics, not a model defect.

## Data Provenance
- Source: Petrobras 3W Dataset (github.com/petrobras/3W), real + simulated
  offshore wells, expert-labeled, 1 Hz sampling.
- Sampled 60 files per event type, 5-minute feature windows.