"""Diagnose the PCK scaling confusion:
Is EARLY scaling (transient) the problem, or ESTABLISHED scaling (fault)?
We keep transient_7 and fault_7 as SEPARATE labels to see which one fails.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("data/threew_features.csv")

# Keep the ORIGINAL stage labels (don't collapse), but simplify:
# everything that isn't scaling-related becomes "other"
def relabel(lbl):
    if lbl == "normal":
        return "normal"
    if lbl == "transient_7":
        return "scaling_EARLY"
    if lbl == "fault_7":
        return "scaling_ESTABLISHED"
    return "other"

data["diag"] = data["label"].apply(relabel)
print("Label counts for diagnosis:")
print(data["diag"].value_counts().to_string())

# Features
feature_cols = [c for c in data.columns
                if c not in ("label", "event", "source_file", "diag")]
X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    lo, hi = X[col].quantile(0.001), X[col].quantile(0.999)
    X[col] = X[col].clip(lo, hi)
X = X.fillna(-999)
X = pd.DataFrame(StandardScaler().fit_transform(X).astype("float32"),
                 columns=X.columns, index=X.index)
y = data["diag"]

# Grouped split (no leakage)
groups = data["source_file"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("\n=== Can the model tell EARLY vs ESTABLISHED scaling from normal? ===\n")
print(classification_report(y_test, preds, zero_division=0))

print("Confusion matrix (rows=truth, cols=guess):")
labels = ["normal", "scaling_EARLY", "scaling_ESTABLISHED", "other"]
labels = [l for l in labels if l in y_test.values]
print(pd.DataFrame(
    confusion_matrix(y_test, preds, labels=labels),
    index=labels, columns=labels,
).to_string())