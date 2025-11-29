import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dvclive import Live
import numpy as np
import pickle



df = pd.read_csv("data/classification_dataset_processed.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=18)

n_estimators = 2
max_depth = 2

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

with Live(save_dvc_exp = True) as live:
    live.log_metric("acuracy", accuracy_score(y_test,y_pred))
    live.log_metric("precision", precision_score(y_test,y_pred))
    live.log_metric("recall", recall_score(y_test,y_pred))
    live.log_metric("f1 sore", f1_score(y_test,y_pred))

    live.log_param("n_estimators", n_estimators)
    live.log_param("max_depth", max_depth)


pickle.dump(rf, open("models/rf_model.pkl","wb"))