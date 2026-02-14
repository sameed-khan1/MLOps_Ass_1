import pandas as pd
import joblib
import os

df = pd.read_csv("features/features.csv")
X = df.drop("Survived", axis=1)

model = joblib.load("models/model.pkl")
predictions = model.predict(X)

os.makedirs("results", exist_ok=True)
pd.DataFrame({"Prediction": predictions}).to_csv(
    "results/predictions.csv", index=False
)
