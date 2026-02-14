import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("features/features.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
