import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error
from abc import ABC, abstractmethod
from scipy import stats
import click

class KNN(ABC):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    @abstractmethod
    def predict(self, X):
        pass

class KNNClassifier(KNN):
    def predict(self, X):
        arr = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X.shape[0]):
                arr[i, j] = np.linalg.norm(self.X[j, :] - X[i, :])
        arr_indices = np.argsort(arr)
        pred_labels = (stats.mode(self.y[item[:self.k]], keepdims=True).mode[0] for item in arr_indices)
        return np.array(list(pred_labels))

class KNNRegression(KNN):
    def predict(self, X):
        arr = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X.shape[0]):
                arr[i, j] = np.linalg.norm(self.X[j, :] - X[i, :])
        arr_indices = np.argsort(arr)
        pred_vals = (np.mean(self.y[item[:self.k]]) for item in arr_indices)
        return np.array(list(pred_vals))


@click.command()
@click.option("--k", default=5)
@click.option("--random_no", default=4848)
@click.option("--t_size", default=0.2)
def main(k, random_no, t_size):
    print("Testing on iris data....")
    df = load_iris()
    classifier = KNNClassifier(k)
    train_X, test_X, y_train, y_test = train_test_split(df.data, df.target, shuffle=True, random_state=random_no, stratify=df.target, test_size=t_size)
    classifier.fit(train_X, y_train)
    y_pred = classifier.predict(test_X)
    print(f"Accuracy from self-made model: {accuracy_score(y_pred, y_test)}")

    model = KNeighborsClassifier(k)
    model.fit(train_X, y_train)
    y_preds = model.predict(test_X)
    print(f"Accuracy from sklearn model: {accuracy_score(y_preds, y_test)}")



    print("Testing on diabetes data....")
    df = load_diabetes()
    regressor = KNNRegression(k)
    train_X, test_X, y_train, y_test = train_test_split(df.data, df.target, shuffle=True, random_state=4848, test_size=t_size)
    regressor.fit(train_X, y_train)
    y_pred = regressor.predict(test_X)
    print(f"RMSE from self-made model: {mean_squared_error(y_pred, y_test) ** 0.5}")
    print(f"MAPE from self-made model: {mean_absolute_percentage_error(y_pred, y_test)}")

    model = KNeighborsRegressor(k)
    model.fit(train_X, y_train)
    y_preds = model.predict(test_X)
    print(f"RMSE from sklearn model: {mean_squared_error(y_preds, y_test) ** 0.5}")
    print(f"MAPE from sklearn model: {mean_absolute_percentage_error(y_preds, y_test)}")

if __name__ == "__main__":
    main()
