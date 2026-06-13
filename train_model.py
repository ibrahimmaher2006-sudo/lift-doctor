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

# 4. Test it on the unseen wells
preds = model.predict(X_test)
print("=== How well it names failures it's never seen ===\n")
print(classification_report(y_test, preds))
print("Confusion matrix (rows=truth, cols=guess):")
print(pd.DataFrame(
    confusion_matrix(y_test, preds, labels=model.classes_),
    index=model.classes_, columns=model.classes_,
))

# 5. Save the trained model for the dashboard to use later
joblib.dump(model, "liftdoctor/model.pkl")
print("\nModel saved to liftdoctor/model.pkl")