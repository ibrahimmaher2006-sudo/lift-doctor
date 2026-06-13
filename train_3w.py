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

# 4. Handle missing sensors: fill NaN with a sentinel the model can use.
#    Real wells lack some sensors; -999 lets the model learn "sensor absent".
X = X.fillna(-999)

# 5. Split train/test, keeping class proportions (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

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
                            target_names=target_names))

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