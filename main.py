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


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * sigmoid(-x)


def reLU(x):
    return np.maximum(0, x)

def d_reLU(x):
    return (x > 0).astype(float)


lambd = 0.01

network = NeuralNetwork(
    layers_dimensions=[30, 50, 100],
    activations=[sigmoid, sigmoid, sigmoid],
    output_activation=lambda x: x,
    output_activation_derivative=lambda x: x,
    activations_derivatives=[d_sigmoid, d_sigmoid, d_sigmoid],
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
