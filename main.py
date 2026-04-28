import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from classes.networks import NeuralNetwork

data = pd.read_csv("data/housing.csv")

t = data["MEDV"]
X = data.drop("MEDV", axis=1)

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)
X_train = X_train.to_numpy()
t_train = t_train.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()
t_test = t_test.to_numpy().reshape(-1, 1)


def tanh(x: np.ndarray) -> np.ndarray:
    # np.tanh já é estável numericamente
    return np.tanh(x)

def d_tanh(x: np.ndarray) -> np.ndarray:
    # derivada: 1 - tanh^2(x)
    t = np.tanh(x)
    return 1 - t**2


lambd = 0.01

network = NeuralNetwork(
    layers_dimensions=[30, 50, 100],
    initialization="xavier",
    activations=[tanh, tanh, tanh],
    output_activation=lambda x: x,
    output_activation_derivative=lambda x: x,
    activations_derivatives=[d_tanh, d_tanh, d_tanh],
    # Mean Squared Error
    error_function=lambda t, y, w: np.linalg.norm(t-y, 'fro')**2 / t.shape[0] + lambd/2 * sum(np.linalg.norm(wi, 'fro')**2 for wi in w)
)

errs = network.fit(X_train, t_train, learning_rate=0.001, epochs=1000, lambd=lambd)

y_test = network.predict(X_test)

print(np.mean((t_test - y_test) ** 2))


plt.title("Error over epochs")
plt.plot(errs)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
