import networkx as nx
import numpy as np
import pandas as pd
import torch
from classes.gnn import GraphConvolutionNetwork
from functions import Identity, LeakyReLU, BinaryCrossEntropyWithLogitsLoss
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200  # usuários

# =========================
# 1. FEATURES
# =========================
features = pd.DataFrame({
    'idade':        np.random.randint(13, 70, n),
    'genero':       np.random.randint(0, 2, n),
    'num_posts':    np.random.randint(0, 500, n),
    'num_amigos':   np.random.randint(1, 300, n),
    'conta_nova':   np.random.randint(0, 2, n),
})

# =========================
# 2. "GROUND TRUTH" (com ruído)
# =========================

# score contínuo (mais realista que regra hard)
score = (
    2.0 * features['conta_nova'] +
    0.01 * features['num_posts'] -
    0.01 * features['num_amigos'] +
    np.random.normal(0, 0.5, n)  # ruído
)

# transforma em probabilidade
prob = 1 / (1 + np.exp(-score))

# amostra binária
fraude_true = (np.random.rand(n) < prob).astype(int)

features['fraude_true'] = fraude_true

print(f"Fraudes reais: {fraude_true.sum()} / {n}")

# =========================
# 3. LABELS OBSERVADOS (parciais)
# =========================

# regra de alta confiança (como você fez)
high_confidence = (
    (features['conta_nova'] == 1) &
    (features['num_posts'] > 300) &
    (features['num_amigos'] < 50)
)

# inicializa como "desconhecido"
fraude_obs = np.full(n, -1)  # -1 = unlabeled

# só rotula alguns positivos com confiança
fraude_obs[high_confidence] = 1

# opcional: adicionar alguns negativos confiáveis
low_risk = (
    (features['conta_nova'] == 0) &
    (features['num_posts'] < 50) &
    (features['num_amigos'] > 150)
)

fraude_obs[low_risk] = 0

features['fraude_obs'] = fraude_obs

# máscara de treino
train_mask = fraude_obs != -1

print(f"Rotulados para treino: {train_mask.sum()} / {n}")

# =========================
# 4. GRAFO
# =========================

cols = ['idade', 'genero', 'num_posts', 'num_amigos', 'conta_nova']
scaler = StandardScaler()

features[cols] = scaler.fit_transform(features[cols])

G = nx.erdos_renyi_graph(n, 0.05, seed=42)
for i in range(n):
    G.nodes[i]['x'] = torch.tensor(features.loc[i, cols].to_numpy(), dtype=torch.float32)


# =========================
# 5. TREINO DO GNN
# =========================

gcn = GraphConvolutionNetwork(
    G=G,
    layers_dimensions=[5, 16, 8, 1],
    output_activation=Identity(),  # logits
    activations=[LeakyReLU(), LeakyReLU()],  # ReLU para as camadas ocultas
    loss_func=BinaryCrossEntropyWithLogitsLoss()
)


T = torch.tensor(fraude_obs, dtype=torch.float32)

epochs = 100

for epoch in range(epochs):
    Y = gcn.forward()
    gcn.backward(Y, T, train_mask)
    gcn.update(learning_rate=0.01)

print(torch.sigmoid(Y))
