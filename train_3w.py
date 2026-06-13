"""Train a 10-class failure classifier on the real 3W dataset."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

EVENT_NAMES = {
    0: "Normal", 1: "BSW increase", 2: "DHSV closure", 3: "Severe slugging",
    4: "Flow instability", 5: "Rapid productivity loss", 6: "Quick PCK restriction",
    7: "PCK scaling", 8: "Hydrate in line", 9: "Event type 9",
}

# 1. Load the windowed features
data = pd.read_csv("data/threew_features.csv")
print(f"Loaded {len(data)} windows")

# 2. Collapse 17 stage-labels into 10 event types
#    'normal' -> 0 ; 'transient_5' or 'fault_5' -> 5 ; etc.
def to_event(label):
    if label == "normal":
        return 0
    return int(label.split("_")[1])   # grabs the number after the underscore

data["event"] = data["label"].apply(to_event)
print(f"\nCollapsed to {data['event'].nunique()} event classes:")
print(data["event"].map(EVENT_NAMES).value_counts().to_string())

# 3. Separate features (X) from target (y)
feature_cols = [c for c in data.columns
                if c not in ("label", "event", "source_file")]
X = data[feature_cols]
y = data["event"]

# 4. Clean AND scale the features — real sensors have wildly different scales
#    (some in single digits, P-TPT in billions). We fix overflow + normalize.
import numpy as np

# 4a. Replace infinities with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# 4b. Clip extreme values per feature (winsorize at 0.1 / 99.9 percentile)
for col in X.columns:
    lo, hi = X[col].quantile(0.001), X[col].quantile(0.999)
    X[col] = X[col].clip(lo, hi)

# 4c. Fill missing-sensor gaps BEFORE scaling (sentinel for "absent")
X = X.fillna(-999)

# 4d. Scale every feature to mean 0, std 1 — fixes float32 overflow AND
#     puts billion-scale pressures and single-digit temps on equal footing.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(
    scaler.fit_transform(X).astype("float32"),   # force float32 now, safely
    columns=X.columns, index=X.index,
)

# 4e. Final safety check in the SAME dtype sklearn will use
assert np.isfinite(X.values).all(), "Still have non-finite values after scaling!"
print("Features cleaned and scaled to comparable ranges (float32-safe).")

# 5. Split by SOURCE FILE so no well appears in both train and test.
#    This prevents data leakage — the model must generalize to unseen wells.
from sklearn.model_selection import GroupShuffleSplit

groups = data["source_file"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Sanity check: confirm no file leaked across the split
train_files = set(data.iloc[train_idx]["source_file"])
test_files = set(data.iloc[test_idx]["source_file"])
print(f"\nTrain files: {len(train_files)}, Test files: {len(test_files)}")
print(f"Overlap (must be 0): {len(train_files & test_files)}")

# 6. Train with class_weight='balanced' to fight imbalance
print("\nTraining Random Forest (this may take a minute)...")
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # rare classes count more
    random_state=0,
    n_jobs=-1,                  # use all CPU cores
)
model.fit(X_train, y_train)

# 7. Evaluate on unseen windows
preds = model.predict(X_test)
present = sorted(y_test.unique())
target_names = [EVENT_NAMES[i] for i in present]

print("\n=== Performance on real offshore well data ===\n")
print(classification_report(y_test, preds, labels=present,
                            target_names=target_names, zero_division=0))

print("Confusion matrix (rows=truth, cols=guess):")
cm = pd.DataFrame(
    confusion_matrix(y_test, preds, labels=present),
    index=[EVENT_NAMES[i] for i in present],
    columns=[EVENT_NAMES[i] for i in present],
)
print(cm.to_string())

# 8. Which sensors matter most? (feature importance)
print("\n=== Top 12 most important features ===")
imp = pd.Series(model.feature_importances_, index=feature_cols)
print(imp.sort_values(ascending=False).head(12).to_string())

# 9. Save the trained model
joblib.dump(model, "liftdoctor/model_3w.pkl")
print("\nModel saved to liftdoctor/model_3w.pkl")