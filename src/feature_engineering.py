import pandas as pd
import os

df = pd.read_csv("data/processed/processed.csv")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
features = df[[
    "Pclass", "Sex", "Age", "Fare",
    "FamilySize", "Embarked_Q", "Embarked_S", "Survived"
]]

os.makedirs("features", exist_ok=True)
features.to_csv("features/features.csv", index=False)
