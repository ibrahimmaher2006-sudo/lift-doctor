"""Train a GRADIENT BOOSTING classifier on 3W — compare vs random forest."""
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

# 1. Load + collapse to 10 event classes (same as before)
data = pd.read_csv("data/threew_features.csv")
def to_event(label):
    return 0 if label == "normal" else int(label.split("_")[1])
data["event"] = data["label"].apply(to_event)
print(f"Loaded {len(data)} windows, {data['event'].nunique()} classes")

# 2. Features — clean + scale (same pipeline)
feature_cols = [c for c in data.columns
                if c not in ("label", "event", "source_file")]
X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    lo, hi = X[col].quantile(0.001), X[col].quantile(0.999)
    X[col] = X[col].clip(lo, hi)
X = X.fillna(-999)
X = pd.DataFrame(StandardScaler().fit_transform(X).astype("float32"),
                 columns=X.columns, index=X.index)
y = data["event"]

# 3. Grouped split by well file (no leakage — same as before)
groups = data["source_file"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"Train files: {len(set(data.iloc[train_idx]['source_file']))}, "
      f"Test files: {len(set(data.iloc[test_idx]['source_file']))}")

# 4. Gradient boosting. It has no class_weight param, so we pass
#    per-sample weights to handle imbalance (rare classes weigh more).
sample_weight = compute_sample_weight("balanced", y_train)

print("\nTraining HistGradientBoosting (may take a couple minutes)...")
model = HistGradientBoostingClassifier(
    max_iter=300,          # number of boosting rounds
    learning_rate=0.1,
    max_depth=None,
    l2_regularization=1.0,
    random_state=0,
)
model.fit(X_train, y_train, sample_weight=sample_weight)

# 5. Evaluate
preds = model.predict(X_test)
present = sorted(y_test.unique())
target_names = [EVENT_NAMES[i] for i in present]

acc = accuracy_score(y_test, preds)
print(f"\n=== GRADIENT BOOSTING — accuracy: {acc:.3f} ===\n")
print(classification_report(y_test, preds, labels=present,
                            target_names=target_names, zero_division=0))

print("Confusion matrix (rows=truth, cols=guess):")
print(pd.DataFrame(
    confusion_matrix(y_test, preds, labels=present),
    index=target_names, columns=target_names,
).to_string())

joblib.dump(model, "liftdoctor/model_3w_gb.pkl")
print("\nModel saved to liftdoctor/model_3w_gb.pkl")
print("\n>>> Compare this accuracy to the Random Forest's 0.83 <<<")