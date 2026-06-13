"""Train gradient boosting on 3W AND save the full inference pipeline."""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

EVENT_NAMES = {
    0: "Normal", 1: "BSW increase", 2: "DHSV closure", 3: "Severe slugging",
    4: "Flow instability", 5: "Rapid productivity loss", 6: "Quick PCK restriction",
    7: "PCK scaling", 8: "Hydrate in line", 9: "Event type 9",
}

# 1. Load + collapse to 10 event classes
data = pd.read_csv("data/threew_features.csv")
def to_event(label):
    return 0 if label == "normal" else int(label.split("_")[1])
data["event"] = data["label"].apply(to_event)
print(f"Loaded {len(data)} windows, {data['event'].nunique()} classes")

feature_cols = [c for c in data.columns
                if c not in ("label", "event", "source_file")]

# 2. Grouped split FIRST (before fitting anything) so the scaler and
#    clip bounds learn ONLY from training data — no test-set leakage.
X_raw = data[feature_cols]
y = data["event"]
groups = data["source_file"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X_raw, y, groups))

X_train_raw = X_raw.iloc[train_idx].copy()
X_test_raw = X_raw.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Train files: {len(set(data.iloc[train_idx]['source_file']))}, "
      f"Test files: {len(set(data.iloc[test_idx]['source_file']))}")

# 3. Learn clipping bounds from TRAINING data only, then apply to both
clip_bounds = {}
for col in feature_cols:
    lo = X_train_raw[col].quantile(0.001)
    hi = X_train_raw[col].quantile(0.999)
    clip_bounds[col] = (lo, hi)

def clean(df, bounds):
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in df.columns:
        lo, hi = bounds[col]
        df[col] = df[col].clip(lo, hi)
    return df.fillna(-999)

X_train_clean = clean(X_train_raw, clip_bounds)
X_test_clean = clean(X_test_raw, clip_bounds)

# 4. Fit scaler on TRAINING data only
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train_clean).astype("float32"),
    columns=feature_cols, index=X_train_clean.index)
X_test = pd.DataFrame(
    scaler.transform(X_test_clean).astype("float32"),   # transform, NOT fit
    columns=feature_cols, index=X_test_clean.index)

# 5. Train
sample_weight = compute_sample_weight("balanced", y_train)
print("\nTraining HistGradientBoosting...")
model = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.1, l2_regularization=1.0, random_state=0)
model.fit(X_train, y_train, sample_weight=sample_weight)

# 6. Evaluate (sanity — should match the ~0.846 you saw)
preds = model.predict(X_test)
present = sorted(y_test.unique())
print(f"\n=== Accuracy: {accuracy_score(y_test, preds):.3f} ===\n")
print(classification_report(y_test, preds, labels=present,
      target_names=[EVENT_NAMES[i] for i in present], zero_division=0))

# 7. SAVE THE FULL PIPELINE as one bundle — everything needed to predict.
bundle = {
    "model": model,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "clip_bounds": clip_bounds,
    "event_names": EVENT_NAMES,
}
joblib.dump(bundle, "liftdoctor/pipeline_3w.pkl")
print("Full pipeline saved to liftdoctor/pipeline_3w.pkl")
print("  (model + scaler + feature order + clip bounds + class names)")