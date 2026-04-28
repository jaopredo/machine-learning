from classes import Algorithm
from typing import Callable, Any
import numpy as np
import heapq


class KNN(Algorithm):
    def __init__(
        self,
        k: int,
        distance_function: Callable[[np.ndarray, np.ndarray], np.floating[Any]]
    ):
        """Initializes a KNN that can be used for regressions and classifications depending on the
        `distance_function` passed. 
    
        Parameters
        ----------
        k: int
            The amount of neighbors to consider when making a prediction
        distance_function: Callable[[np.ndarray, np.ndarray], np.floating[Any]]
            The function used to calculate the distance between two points
        """
        self.k = k
        self.distance_function = distance_function
        self.X = None
        self.t = None
        self.labels = None
    
    def fit(self, X: np.ndarray, t: np.ndarray):
        """Fits the KNN model to the given data. For KNN, this simply means storing the training data.

        Args:
            X (np.ndarray): The samples to fit
            t (np.ndarray): The labels of the training set
        """
        self.X = X
        self.t = t
        self.labels = np.unique(t)
    
    def predict(self, p: np.ndarray):
        """Predicts the label of a given point `p` based on the samples `X` and their
        labels `t` using the KNN algorithm.

        Args:
            X (np.ndarray): The samples to predict
            t (np.ndarray): The labels of the training set
            p (np.ndarray): The samples of the training set
        """
        if p.shape[-1] != self.X.shape[1]:
            raise ValueError("The given point does not have the required number of features")
        
        distances = []
        # Distance, Index, Label of the dataset point compared to the given point
    
        for j, sample in enumerate(self.X):
            distance = self.distance_function(sample, p)
            heapq.heappush(distances, (distance, j, self.t[j]))
        
        nearest_neighbors = heapq.nsmallest(self.k, distances)
        
        classification = np.zeros_like(self.labels, dtype=int)
        for nearest_neighbor in nearest_neighbors:
            classification[np.where(self.labels == nearest_neighbor[2])[0][0]] += 1
        return np.argmax(classification)
