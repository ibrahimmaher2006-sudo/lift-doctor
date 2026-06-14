# 🛢️ Lift Doctor — Offshore Well Failure Diagnosis

An AI system that reads real offshore oil well sensor data, detects which of
9 failure types is developing, identifies when it began, and explains the
diagnosis in plain language — with confidence-aware honesty about its limits.

Built on the **Petrobras 3W dataset** (real, expert-labeled wells).

---

## What it does

Given a well's multi-sensor time-series, Lift Doctor:
- **Classifies** the operating state: normal or one of 9 failure types
- **Pinpoints onset** — when the fault first appeared (early-detection focus)
- **Cites evidence** — which sensors moved, and by how much
- **Recommends action** — what an engineer should check
- **Flags uncertainty** — says "uncertain" rather than guessing, and warns
  when a fault resembles another it's known to confuse

## The 9 failure types
Normal, BSW (water-cut) increase, DHSV valve closure, severe slugging,
flow instability, rapid productivity loss, quick choke restriction,
choke scaling, hydrate formation, and a generic event class.

## How it works

1. **Feature extraction** — each well is sliced into 5-minute windows;
   per-sensor mean, standard deviation, and trend (slope) are computed.
2. **Model** — a gradient-boosted classifier (HistGradientBoosting),
   class-weighted to handle severe class imbalance.
3. **Leakage-safe validation** — train/test split is grouped *by well*, so
   the model is always evaluated on wells it has never seen.
4. **Persistence-based diagnosis** — a fault must persist for several
   consecutive windows to be reported, suppressing scattered false alarms.
5. **Explanation layer** — domain-knowledge fault profiles turn the model's
   output and the sensor evidence into a plain-English assessment.

## Honest performance

- **~85% accuracy** classifying failures in **unseen wells**
- Strong on distinctive faults (BSW, DHSV, slugging, choke restriction,
  hydrate): F1 > 0.90
- Weaker where failures are physically ambiguous (slugging vs flow
  instability) or rare in the data (see Limitations)

> Note: an early version scored 99% — which turned out to be **data leakage**
> (windows from the same well in both train and test). Fixing it dropped the
> honest score to 68%, which was then improved to ~85% through more data,
> feature scaling, and a better algorithm. Every gain is leakage-free.

## Known limitations

- **Established choke scaling** is rarely detected — the dataset has ~28,900
  early-scaling windows but only ~440 established ones, too few to learn.
  Accepted as a data limitation rather than fixed artificially.
- **Slugging ↔ flow instability** are sometimes confused — they are genuinely
  similar phenomena. The explanation layer flags this explicitly.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn streamlit pyarrow joblib

# Build the dataset, train the model, then launch the dashboard
python -m liftdoctor.threew          # extract features from 3W files
python train_3w_gb.py                # train + save the pipeline
streamlit run app_3w.py              # interactive dashboard
```

## Project structure

- `liftdoctor/threew.py` — 3W loader + feature extraction
- `train_3w_gb.py` — trains and saves the full inference pipeline
- `liftdoctor/predict.py` — diagnosis logic (onset detection, persistence filter)
- `liftdoctor/explain.py` — plain-English explanation layer
- `app_3w.py` — Streamlit dashboard
- `MODEL_NOTES.md` — detailed performance and limitations

## Data

Petrobras 3W Dataset — github.com/petrobras/3W — real and simulated offshore
well events, expert-labeled, 1 Hz sampling.

---

*Built as a learning-to-production project: from a simulated prototype to a
leakage-validated classifier on real field data, with an emphasis on honest
measurement and transparency about model limits.*
