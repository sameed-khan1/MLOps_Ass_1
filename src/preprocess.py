import pandas as pd
import os

df = pd.read_csv("data/raw/titanic.csv")

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/processed.csv", index=False)
