from classes.mlp import MultilayerPerceptron
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from classes.gcn import GraphConvolutionNetwork
from functions import Identity, LeakyReLU, MeanSquaredError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")


data = pd.read_csv("data/housing.csv")
t = data['MEDV']
X = data.drop(columns=['MEDV'])

t = torch.tensor(t.values, dtype=torch.float32, device=device).unsqueeze(1)
X = torch.tensor(X.values, dtype=torch.float32)

scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32, device=device)

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

model = MultilayerPerceptron(
    layers_dimensions=[X.shape[1], 64, 32, 16, 1],
    output_activation=Identity(),
    activations=[LeakyReLU(), LeakyReLU(), LeakyReLU()],
    loss_func=MeanSquaredError(),
    device=device
)


epochs = 1000
loss_history = []

for epoch in range(epochs):
    y_train = model.forward(X_train)
    print(y_train.shape)
    print(t_train.shape)
    model.backward(y_train, t_train)
    model.update(learning_rate=0.01)

    loss = model.loss_func(y_train, t_train).item()
    loss_history.append(loss)


y_test = model.forward(X_test)
test_loss = model.loss_func(y_test, t_test).item()
print(f"Test Loss: {test_loss:.4f}")


plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_history)
plt.show()
