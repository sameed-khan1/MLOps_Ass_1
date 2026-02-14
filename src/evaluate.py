import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("features/features.csv")
X = df.drop("Survived", axis=1)
y = df["Survived"]

model = joblib.load("models/model.pkl")
y_pred = model.predict(X)

with open("results/evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y, y_pred):.4f}\n")
    f.write(f"Precision: {precision_score(y, y_pred):.4f}\n")
    f.write(f"Recall: {recall_score(y, y_pred):.4f}\n")
    f.write(f"F1-score: {f1_score(y, y_pred):.4f}\n")
