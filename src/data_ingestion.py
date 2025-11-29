import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dvclive import Live
import numpy as np


import yaml

df = pd.read_csv("data/classification_dataset.csv")
with open("params.yaml","r") as file:
    params = yaml.safe_load(file)
# print(params)

np.random.seed(params["data_ingestion"]["seed"])
random_noise = np.random.randint(1,100,size=df.shape[0])
# print(random_noise)

df["IQ"] += random_noise

df.to_csv("data/classification_dataset_processed.csv")
