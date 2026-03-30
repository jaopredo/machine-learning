from classes.networks import NeuralNetwork
import numpy as np

network = NeuralNetwork(
    layers_dimensions=[2, 3, 1],
    activations=[lambda x: x, lambda x: x],
    activations_derivatives=[lambda x: np.ones_like(x), lambda x: np.ones_like(x)],
    error_function=lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
)


X = np.random.rand(100, 2)
t = np.random.rand(100, 1)


network.fit(X, t, learning_rate=0.01, epochs=100)
