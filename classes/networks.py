from classes import Algorithm
from typing import Callable, Any, Literal
import numpy as np


class NeuralNetwork(Algorithm):
    def __init__(
        self,
        layers_dimensions: list[int],
        output_activation: Callable[[np.ndarray], np.ndarray],
        output_activation_derivative: Callable[[np.ndarray], np.ndarray],
        activations: list[Callable[[np.ndarray], np.ndarray]],
        activations_derivatives: list[Callable[[np.ndarray], np.ndarray]],
        error_function: Callable[[np.ndarray, np.ndarray, list[np.ndarray]], np.floating[Any]],
        initialization: Literal["xavier", "he"] = "xavier",
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
        error_function: Callable[[np.ndarray, np.ndarray, list[np.ndarray]], np.floating[Any]]
            The error function used to calculate the error history on the training method
        """
        self.h = activations + [output_activation]
        self.dh = activations_derivatives + [output_activation_derivative]
        self._layers_dimensions = layers_dimensions
        self._error_function = error_function
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.A: list[np.ndarray] = []
        self.Z: list[np.ndarray] = []
        self.initialization = initialization


    def predict(
        self,
        X: np.ndarray
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
        _, Z = self.forward_propagation(X)
        return Z[-1]

    def forward_propagation(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
        W: list[np.ndarray] = self.W
        b: list[np.ndarray] = self.b
        A: list[np.ndarray] = []
        Z: list[np.ndarray] = []
        P = len(self._layers_dimensions)
        h = self.h

        for p in range(P+1):
            ONES = np.ones((X.shape[0], 1))
            if p == 0:
                A.append(X @ W[p] + ONES @ b[p])
                Z.append(h[p](A[p]))
            else:
                A.append(Z[p-1] @ W[p] + ONES @ b[p])
                Z.append(h[p](A[p]))
        
        return A, Z
    
    def back_propagation(
        self,
        X: np.ndarray,
        T: np.ndarray,
        learning_rate: float = 0.01,
        lambd: float = 0,
        max_norm: float = 5.0
    ):
        """This function calculates the backpropagation of the data passed as an argument, updating the weights and bias

        Args:
            X (np.ndarray): The data that will be used for the backpropagation
            T (np.ndarray): The vector with the correct values for each prediction
            learning_rate (float): How much the gradient will move to learn the weights
            epochs (int): How many epochs will it take for the training to end
            lambd (float): Regularization term

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists, the first one is the list of weight updates for each layer,
            and the second one is the list of bias updates for each layer
        """
        W = self.W
        b = self.b
        A = self.A
        Z = self.Z
        dh = self.dh
        P = len(self._layers_dimensions)
        D: list[np.ndarray] = [np.zeros_like([0])] * (P+1)
        dW = [np.zeros_like([0])] * (P+1)
        db = [np.zeros_like([0])] * (P+1)

        # Calculating the gradients for each layer
        for layer in range(P, 0, -1):
            # If I'm on the first layer, I'll apply the specific formula for the
            # output layer, otherwise, I'll apply the formula for the hidden layers
            if layer == P:
                # Remembering that Z[P] = Y (The predicitons)
                D[layer] = (Z[layer] - T) * dh[layer](A[layer])
            else:
                D[layer] = (D[layer+1] @ W[layer+1].T) * dh[layer](A[layer])
            
            dW[layer] = np.zeros(W[layer].shape)
            dW[layer] += Z[layer-1].T @ D[layer]
            dW[layer] /= X.shape[0]
            dW[layer] += lambd * W[layer]
            db[layer] = np.mean(D[layer], axis=0, keepdims=True)
        
        # After I calculated all my gradients, I need to update my weights and bias
        for layer in range(P+1):
            norm = np.linalg.norm(dW[layer])
            if norm > max_norm:
                dW[layer] = dW[layer] / norm * max_norm
            W[layer] -= learning_rate * dW[layer]
            b[layer] -= learning_rate * db[layer]

    def fit(self,
        X: np.ndarray,
        T: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        lambd: float = 0,
        max_norm: float = 5.0
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
        # First, I need to initialize my weights and bias
        # Because W and b are lists of arrays, I will separate them for reading purposes
        W = self.W
        b = self.b
        P = len(self._layers_dimensions)
        errs = []

        for i in range(P+1):
            if i == 0:
                fan_in = X.shape[1]
                fan_out = self._layers_dimensions[i]
            elif i == P:
                fan_in = self._layers_dimensions[i-1]
                fan_out = T.shape[1]
            else:
                fan_in = self._layers_dimensions[i-1]
                fan_out = self._layers_dimensions[i]

            if self.initialization == "xavier":
                std = np.sqrt(1 / fan_in)
            elif self.initialization == "he":
                std = np.sqrt(2 / fan_in)
            else:
                raise ValueError("Unknown initialization")

            W.append(np.random.normal(0, std, (fan_in, fan_out)))
            
            # 🔥 Bias: melhor usar zero (mais estável)
            b.append(np.zeros((1, fan_out)))

        for epoch in range(epochs):
            # Make the forward propagation, obtaining A and Z for each layer
            self.A, self.Z = self.forward_propagation(X)
            # Now I make the backpropagation, calculating the error and updating the weights and bias
            errs.append(self._error_function(T, self.Z[-1], self.W))
            self.back_propagation(X, T, learning_rate, lambd, max_norm)
        
        return errs
