"""Trains a classifier to name the failure from a well's signature."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the labeled dataset
data = pd.read_csv("data/training.csv")
X = data.drop(columns=["label"])   # the 12 signature numbers
y = data["label"]                  # the truth

# 2. Split: train on 80%, test on 20% it has NEVER seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# 3. Train the classifier
model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

# 4. Test it on the unseen wells — WITH a confidence threshold
CONFIDENCE = 0.55   # below this, the model says "uncertain" instead of guessing

probs = model.predict_proba(X_test)
top_prob = probs.max(axis=1)               # how confident the top guess is
raw_preds = model.classes_[probs.argmax(axis=1)]
preds = [p if c >= CONFIDENCE else "uncertain"
         for p, c in zip(raw_preds, top_prob)]

print("=== Performance WITH confidence threshold ===\n")

# How often does it abstain, and is it right when it commits?
committed = [(t, p) for t, p in zip(y_test, preds) if p != "uncertain"]
n_uncertain = preds.count("uncertain")
n_committed = len(committed)
correct = sum(1 for t, p in committed if t == p)

print(f"Confidence threshold: {CONFIDENCE}")
print(f"Abstained (said 'uncertain'): {n_uncertain}/{len(preds)} "
      f"({100*n_uncertain/len(preds):.0f}%)")
print(f"Accuracy WHEN it committed: {correct}/{n_committed} "
      f"({100*correct/n_committed:.0f}%)\n")

print("Confusion matrix including 'uncertain' column:")
labels = list(model.classes_) + ["uncertain"]
print(pd.DataFrame(
    confusion_matrix(y_test, preds, labels=labels),
    index=labels, columns=labels,
))

# 5. Save the trained model for the dashboard to use later
joblib.dump(model, "liftdoctor/model.pkl")
print("\nModel saved to liftdoctor/model.pkl")