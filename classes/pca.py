import torch
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X: torch.Tensor):
        # Center the data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        n = X_centered.shape[0]

        # Compute covariance matrix
        cov_matrix = torch.mm(X_centered.t(), X_centered) / (n)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        total_variance = sum(eigenvalues)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = torch.argsort(eigenvalues, descending=True)

        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = sum(eigenvalues[:self.n_components]) / total_variance

    def transform(self, X: torch.Tensor):
        return torch.mm(X, self.components) 
