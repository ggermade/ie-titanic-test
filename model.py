# The imports at the beginning!
import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression

def train_and_persist():
    df = pd.read_csv('titanic.csv', index_col="PassengerId")
    # print(df.head())

    X = df.drop(columns=["Survived"]).select_dtypes(include=np.number).dropna(how="any")
    y = df.loc[X.index, "Survived"]
    # print(X.head())

    clf = LogisticRegression()
    clf.fit(X, y)
    joblib.dump(clf, "titanic.joblib")

def predict(pclass, age, sibsp, parch, fare):
    clf = joblib.load("titanic.joblib")
    X_new = pd.DataFrame([[
        pclass, age, sibsp, parch, fare
    ]])

    y_pred = clf.predict(X_new)
    return bool(y_pred[0])
