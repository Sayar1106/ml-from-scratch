import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class KNN(ABC):
    def __init__(self, k):
        self.k = k

    @abstractmethod
    def fit(X, y):
        pass

    @abstractmethod
    def predict(X):
        pass

class KNNClassifier(KNN):
    pass
