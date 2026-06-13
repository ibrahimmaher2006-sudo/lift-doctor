"""Is the bug in the MODEL or in prediction-time feature computation?
Compare: model on training-CSV features  vs  model on freshly-computed features.
"""
import pandas as pd
import numpy as np
import joblib

bundle = joblib.load("liftdoctor/pipeline_3w.pkl")
model, scaler, cols = bundle["model"], bundle["scaler"], bundle["feature_cols"]
clip = bundle["clip_bounds"]

# Load the SAME features used in training
data = pd.read_csv("data/threew_features.csv")
def to_event(l): return 0 if l == "normal" else int(l.split("_")[1])
data["event"] = data["label"].apply(to_event)

# Take 500 random rows, preprocess EXACTLY like training, predict
sample = data.sample(500, random_state=1)
X = sample[cols].replace([np.inf, -np.inf], np.nan)
for c in cols:
    lo, hi = clip[c]
    X[c] = X[c].clip(lo, hi)
X = X.fillna(-999)
Xs = pd.DataFrame(scaler.transform(X), columns=cols)

preds = model.predict(Xs)
actual = sample["event"].values
acc = (preds == actual).mean()

print(f"Model accuracy on training-CSV features (through saved pipeline): {acc:.3f}")
print("\nIf this is ~0.85+, the MODEL + SCALER are fine,")
print("and the bug is in predict.py's feature computation from raw files.")
print("If this is also low, the bug is in the saved model/scaler themselves.\n")

# Show what it predicts vs actual for a few
print("Sample predictions (actual -> predicted):")
for a, p in list(zip(actual, preds))[:15]:
    print(f"  {bundle['event_names'][a]:25} -> {bundle['event_names'][p]}")