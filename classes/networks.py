from classes import Algorithm
from typing import Callable
import numpy as np


class NeuralNetwork(Algorithm):
    def __init__(
        self,
        layers_dimensions: list[int],
        activations: list[Callable[[np.ndarray], np.ndarray]],
        activations_derivatives: list[Callable[[np.ndarray], np.ndarray]],
        error_function: Callable[[np.ndarray, np.ndarray], float]
    ):
        """Initializes a NeuralNetwork that can be used for regressions and classifications depending on the
        `activation` functions passed. 
    
        Parameters
        ----------
        layers_dimensions: list[int]
            The list containing the amount of features per layer (Do not include the BIAS)
        activations : list[Callable[[np.ndarray], np.ndarray]]
            List containing the activation functions used in the hidden layers, where the i-th function of
            the list is assigned to the i-th layer
        activations_derivatives: list[Callable[[np.ndarray], np.ndarray]]
            List containing the derivatives of each function passed in the `activations` argument. If
            `activations` has `n` entries, this argument must have `n-1` as we don't use the derivative
            of the last layer (output), for that, the target column must be adapted
        error_function: Callable[[np.ndarray, np.ndarray], float]
            The error function used to calculate the error history on the training method
        """
        
        self.h = activations
        self.dh = activations_derivatives
        self._layers_dimensions = layers_dimensions
        self._error_function = error_function

        if len(activations) != len(layers_dimensions) - 1:
            raise ValueError("The amount of activation functions must be equal to the amount of layers minus one")
        if len(activations_derivatives) != len(activations):
            raise ValueError("The amount of activation derivatives must be equal to the amount of activation functions")


    def predict(self,
        X: np.ndarray,
        func_result: Callable[[np.ndarray], None] = None,
        func_weighted_sum: Callable[[np.ndarray], None] = None
    ) -> np.ndarray:
        """Function that predicts a given X according to the weights stored at the moment

        Args
        ----------
        X: np.ndarray
            Data that will be predicted
        func_result (optional): Callable[[np.ndarray], None]
            This parameters receives a function that will be executed every end of loop
            and pass the activation result as an argument
        func_weighted_sum (optional): Callable[[np.ndarray], None]
            This parameters receives a function that will be executed every end of loop
            and pass the weighted sum before activation as an argument

        Returns:
            np.ndarray: The prediction based on the actual weights
        """

    def forward_propagation(self, X: np.ndarray) -> list[np.ndarray]:
        """This function calculates the forward propagation of the data passed as an argument

        Args
        ----------
        X: np.ndarray
            The data that will be used for the forward propagation

        Returns
        ----------
        list[np.ndarray]:
            A list containing the activation result of each layer, where the i-th entry is the result of the i-th layer
        """
        W = self._weights
        b = self._biases
        h = self._activations
        A = [X @ W[0] + b[0] @ np.ones((1, W[0].shape[1]))]
        Z = [h[0](A[0])]

        for i in range(len(self._layers_dimensions) - 1):
            A.append(Z[i] @ W[i + 1] + b[i + 1] @ np.ones((1, W[i + 1].shape[1])))
            Z.append(h[i](A[i + 1]))
        
        return Z, A


    def fit(self,
        X: np.ndarray,
        t: np.ndarray,
        learning_rate: float,
        epochs: int,
        lambd: float = 0
    ) -> list[float]:
        """This function trains the weights based on the data passed and the target given

        Args
        ----------
        X: np.ndarray
            The data that will be used for training
        t: np.ndarray
            The vector with the correct values for each prediction
        learning_rate: float
            How much the gradient will move to learn the weights
        epochs: int
            How many epochs will it take for the training to end
        lambd: float
            Regularization term
        
        Returns
        ----------
        list[float]:
            A list containing the error for each epoch
        """
        error_history = []

        # Initializing the weights
        # I have that X in RR^(N x D), so the first weight, because I want to have
        W = [np.random.normal(0, 1, (X.shape[1], self._layers_dimensions[0]))]
        # I'll sum the biases over all datapoints, so I'll have a bias for each layer with the same
        # amount of rows as X, so I can sum it to the weighted sum
        b = [np.random.normal(0, 1, (X.shape[0], 1))] * len(self._layers_dimensions)

        for i in range(len(self._layers_dimensions) - 1):
            W.append(
                np.random.normal(
                    0,
                    1,
                    (self._layers_dimensions[i], self._layers_dimensions[i + 1])
                )
            )
        
        Z, A = self.forward_propagation(X)
