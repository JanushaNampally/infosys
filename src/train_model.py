
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "../dataset.csv"
MODEL_PATH = "../models/activity_model.pkl"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# IMPORTANT: Create fake video groups (since we didn't store video names)
# We simulate grouping by chunking dataset
df["group"] = df.index // 300  # each 300 frames treated as one video group

X = df.drop(["label", "group"], axis=1)
y = df["label"]
groups = df["group"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X, y_encoded, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print("Training model...")

model = RandomForestClassifier(
    n_estimators=50,       # reduced trees
    max_depth=10,          # restrict complexity
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

os.makedirs("../models", exist_ok=True)
joblib.dump((model, label_encoder), MODEL_PATH)

print("\nModel saved successfully!")