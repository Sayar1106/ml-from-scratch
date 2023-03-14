import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from random import random 

class KNN(ABC):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X

    @abstractmethod
    def predict(self, X):
        pass

class KNNClassifier(KNN):
    def predict(self, X):
        distance_list = np.linalg.norm(self.X - X)
        print(distance_list)

    

if __name__ == "__main__":
    df = load_iris()
    classifier = KNNClassifier(5)
    idx = random()
    train_X, test_X, y_train, y_test = train_test_split(df, shuffle=True, random_state=4848, test_size=0.2)
    classifier.fit(train_X)
    classifier.predict(test_X)