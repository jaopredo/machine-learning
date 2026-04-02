from abc import ABC, abstractmethod
from numpy import ndarray


class Algorithm(ABC):
    @abstractmethod
    def fit(self, X: ndarray, T: ndarray) -> list[float]:
        """Fits the model onto the data

        Args:
            X (ndarray): NxD array of N samples and D features
            T (ndarray): Nx1 array of N target values
        """
        pass
    
    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """Predicts the target values for the given data

        Args:
            X (ndarray): NxD array of N samples and D features
        """
        pass