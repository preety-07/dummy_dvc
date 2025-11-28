import pandas as pd


df = pd.read_csv("data/classification_dataset.csv")
print(df.head())
print(df["Placed"].value_counts())
