import sys
from pathlib import Path

root = Path().resolve()  # aponta para machine-learning/
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from classes.regression import LogisticRegression
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X = load_breast_cancer().data
y = load_breast_cancer().target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
t_train = torch.tensor(t_train, dtype=torch.float32, device=device)

X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
t_test = torch.tensor(t_test, dtype=torch.float32, device=device)

model = LogisticRegression(device=device)
model.fit(X_train, t_train, mode='maximum_likelihood', lr=0.01, epochs=1000)

predictions = model.predict(X_test)
accuracy = (predictions == t_test).float().mean()

print(f"Acurácia: {accuracy:.4f}")

# Curva ROC
from sklearn.metrics import roc_curve, auc
probabilities = model.predict(X_test, proba=True).cpu().numpy()
fpr, tpr, _ = roc_curve(t_test.cpu().numpy(), probabilities)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
