from typing import Any, Callable
import heapq

import numpy as np
import torch
import torch.nn as nn


class KNN(nn.Module):
    def __init__(
        self,
        k: int,
        distance_function: Callable[[np.ndarray, np.ndarray], np.floating[Any]],
        X: np.ndarray | torch.Tensor | None = None,
        t: np.ndarray | torch.Tensor | None = None,
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
        super().__init__()
        self.k = k
        self.distance_function = distance_function
        self.X = None
        self.t = None
        self.labels = None
        if X is not None or t is not None:
            if X is None or t is None:
                raise ValueError("Both X and t must be provided together")
            self.set_data(X, t)

    @staticmethod
    def _to_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)
    
    def set_data(self, X: np.ndarray | torch.Tensor, t: np.ndarray | torch.Tensor):
        self.X = self._to_numpy(X)
        self.t = self._to_numpy(t)
        self.labels = np.unique(self.t)
    
    def _predict_single(self, p: np.ndarray):
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

    def forward(self, p: np.ndarray | torch.Tensor):
        """Predicts the label of a given point `p` based on the samples `X` and their
        labels `t` using the KNN algorithm.

        Args:
            X (np.ndarray): The samples to predict
            t (np.ndarray): The labels of the training set
            p (np.ndarray): The samples of the training set
        """
        if self.X is None or self.t is None or self.labels is None:
            raise ValueError("Model has no training data. Provide X and t in the constructor or call set_data().")

        p = self._to_numpy(p)

        if p.ndim == 1:
            return self._predict_single(p)
        if p.ndim == 2:
            return np.asarray([self._predict_single(point) for point in p])

        raise ValueError("The given point must be a 1D sample or a 2D batch of samples")
