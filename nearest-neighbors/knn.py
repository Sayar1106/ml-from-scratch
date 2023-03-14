import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from random import random 
from scipy import stats

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
        pred_labels = (stats.mode(self.y[item[:self.k]]).mode[0] for item in arr_indices)
        return list(pred_labels)

        

    

if __name__ == "__main__":
    df = load_iris()
    classifier = KNNClassifier(5)
    idx = random()
    train_X, test_X, y_train, y_test = train_test_split(df.data, df.target, shuffle=True, random_state=4848, test_size=0.2)
    classifier.fit(train_X, y_train)
    y_pred = classifier.predict(test_X)
    print(y_test)
    print(y_pred)
    # print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
