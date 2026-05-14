import numpy as np
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian


class LinearRegression(nn.Module):
    def __init__(
        self,
        X: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        mode: Literal['ols', 'sgd'] = 'ols',
        c=1e-8,
        lr=0.01,
        n_iterations=1000,
        stop_threshold=1e-6,
        batch_size=32,
        max_norm=1.0,
        device:torch.device | None = None
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")

        self.theta = None
        self.X = None
        self.t = None
        if X is not None or t is not None:
            if X is None or t is None:
                raise ValueError("Both X and t must be provided together")

            self._fit(
                X,
                t,
                mode=mode,
                c=c,
                lr=lr,
                n_iterations=n_iterations,
                stop_threshold=stop_threshold,
                batch_size=batch_size,
                max_norm=max_norm,
            )

    @staticmethod
    def _add_bias(X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)
        return torch.cat([torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype), X], dim=1)

    def ols(self, c=1e-8):
        """Calculates the optimal parameters Theta using the Normal Equation."""
        identity = torch.eye(self.X.shape[1], device=self.device, dtype=self.X.dtype)
        return torch.linalg.solve(self.X.T @ self.X + c * identity, self.X.T @ self.t)

    def sgd(self, learning_rate=0.01, n_iterations=1000, stop_threshold=1e-6, batch_size=32, max_norm=1.0):
        """Calculates the optimal parameters Theta using Stochastic Gradient Descent."""
        n, d = self.X.shape
        theta = torch.randn(d, device=self.device, dtype=self.X.dtype)
        previous_theta = None
        for _ in range(n_iterations):
            previous_theta = theta.clone()
            for _ in range(batch_size):
                i = torch.randint(n, (1,), device=self.X.device).item()
                xi = self.X[i:i+1]
                ti = self.t[i:i+1]
                gradients = 2/n * xi.T @ (xi @ theta - ti)

                norm = torch.norm(gradients)
                if norm.item() > max_norm:
                    gradients = gradients / norm * max_norm
                theta -= learning_rate * gradients.flatten()
                
            if torch.norm(theta - previous_theta).item() < stop_threshold:
                return theta
        return theta

    def _fit(
            self,
            X: torch.Tensor,
            t: torch.Tensor,
            mode: Literal['ols', 'sgd'] = 'ols',
            c=1e-8,
            lr=0.01,
            n_iterations=1000,
            stop_threshold=1e-6,
            batch_size=32,
            max_norm=1.0
        ):
        """Fits the Linear Regression model to the given data by calculating the optimal parameters Theta
        using the Normal Equation.

        Args:
            X (torch.Tensor): The samples to fit
            t (torch.Tensor): The labels of the training set
            mode (Literal['ols', 'sgd']): The method used to calculate the optimal parameters.
            'ols' for Ordinary Least Squares, 'sgd' for Stochastic Gradient Descent.
        """
        # Add bias term to the features
        X_b = self._add_bias(X)
        self.X = X_b
        self.t = t

        if mode == 'ols':
            self.theta = self.ols(c=c)
        elif mode == 'sgd':
            self.theta = self.sgd(
                learning_rate=lr,
                n_iterations=n_iterations,
                stop_threshold=stop_threshold,
                batch_size=batch_size,
                max_norm=max_norm
            )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predicts the labels for the given samples using the learned parameters Theta.

        Args:
            X (torch.Tensor): The samples to predict
        Returns:
            torch.Tensor: The predicted labels for the given samples
        """
        if self.theta is None:
            raise ValueError("Model is not fitted yet. Provide training data in the constructor.")

        X_t = self._to_tensor(X, device=self.device, dtype=self.theta.dtype)
        X_b = self._add_bias(X_t)
        predictions = X_b @ self.theta

        return predictions


class LogisticRegression(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.theta = None
        self.X = None
        self.t = None
        self.mode: Literal['maximum_posteriori', 'laplace']=None
        self.loss_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def maximum_posterirori(self, mean=None, covariance_matrix=None, epochs=10000, lr=0.01):
        n, d = self.X.shape
        theta = torch.randn(d, dtype=self.X.dtype, device=self.X.device, requires_grad=True)
        mean = torch.zeros(d, dtype=self.X.dtype, device=self.X.device) if mean is None else mean

        optimizer = torch.optim.SGD([theta], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            # logits
            z = self.X @ theta

            # log-likelihood (negativa da BCE porque queremos maximizar)
            log_likelihood = -F.binary_cross_entropy_with_logits(z, self.t, reduction='sum')

            # log-prior gaussiano
            diff = theta - mean
            prior_term = -0.5 * diff @ torch.linalg.solve(covariance_matrix, diff)

            # log-posterior
            log_posterior = log_likelihood + prior_term

            # queremos maximizar → minimizar o negativo
            loss = -log_posterior
            loss.backward()

            self.loss_history.append(loss.item())

            optimizer.step()

        return theta.detach()
    
    def neg_log_posterior(self, theta, mean, covariance_matrix):
        z = self.X @ theta
        
        nll = F.binary_cross_entropy_with_logits(z, self.t, reduction='sum')
        
        diff = theta - mean
        prior = 0.5 * diff @ torch.linalg.solve(covariance_matrix, diff)
        
        return nll + prior

    def laplace(self, mean, covariance_matrix):
        theta_map = self.theta.clone().detach().requires_grad_(True)

        H = hessian(
            lambda th: self.neg_log_posterior(th, mean, covariance_matrix),
            theta_map
        )

        # regularização numérica (importante)
        eps = 1e-6
        H = H + eps * torch.eye(H.shape[0], device=H.device)

        cov = torch.linalg.inv(H)

        self.cov = cov
    
    def variational(self, mean, covariance_matrix, epochs=1000, lr=0.01, M=10):
        n, d = self.X.shape
        device = self.X.device
        dtype = self.X.dtype

        # parâmetros variacionais
        mu = torch.randn(d, dtype=dtype, device=device, requires_grad=True)
        rho = torch.zeros(d, dtype=dtype, device=device, requires_grad=True)  # log sigma

        optimizer = torch.optim.Adam([mu, rho], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            sigma = torch.exp(rho)

            elbo = 0.0

            for _ in range(M):
                eps = torch.randn(d, dtype=dtype, device=device)
                theta = mu + sigma * eps  # reparametrização

                # log-likelihood
                z = self.X @ theta
                log_likelihood = -F.binary_cross_entropy_with_logits(z, self.t, reduction='sum')

                # log-prior
                diff = theta - mean
                log_prior = -0.5 * diff @ torch.linalg.solve(covariance_matrix, diff)

                # log q(theta)
                log_2pi = torch.log(torch.tensor(2 * torch.pi, dtype=dtype, device=device))
                log_q = -0.5 * torch.sum(((theta - mu) / sigma) ** 2 + 2 * rho + log_2pi)

                elbo += log_likelihood + log_prior - log_q

            elbo = elbo / M

            loss = -elbo
            loss.backward()
            self.loss_history.append(loss.item())
            optimizer.step()

        self.theta = mu.detach()
        self.sigma = torch.exp(rho).detach()
    
    def fit(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        mode: Literal['maximum_posteriori', 'laplace', 'variational']='maximum_posteriori',
        mean=None, covariance_matrix=None, epochs=10000, lr=0.01
    ):
        X = X.to(self.device)
        t = t.to(self.device)
        _, d = X.shape
        theta = torch.randn(d, device=self.device, dtype=X.dtype)
        self.mode = mode
        self.loss_history = []

        self.theta = theta
        self.X = X
        self.t = t

        if mode == 'maximum_posteriori':
            if covariance_matrix is None:
                raise ValueError("Covariance matrix must be provided for maximum posteriori mode.")
            if mean is None:
                mean = torch.zeros(d, dtype=X.dtype, device=self.device)
            else:
                mean = mean.to(self.device, dtype=X.dtype)
            covariance_matrix = covariance_matrix.to(self.device, dtype=X.dtype)
            
            # Getting argmax of the posteriori: log(p(D|theta)) + log(p(theta))
            self.theta = self.maximum_posterirori(
                mean=mean,
                covariance_matrix=covariance_matrix,
                epochs=epochs,
                lr=lr
            )

        elif mode == 'laplace':
            if covariance_matrix is None:
                raise ValueError("Covariance matrix must be provided for laplace mode.")
            if mean is None:
                mean = torch.zeros(d, dtype=X.dtype, device=self.device)
            else:
                mean = mean.to(self.device, dtype=X.dtype)
            covariance_matrix = covariance_matrix.to(self.device, dtype=X.dtype)

            self.theta = self.maximum_posterirori(
                mean=mean,
                covariance_matrix=covariance_matrix,
                epochs=epochs,
                lr=lr
            )

            self.laplace(mean, covariance_matrix)

        elif mode == 'variational':
            if covariance_matrix is None:
                raise ValueError("Covariance matrix must be provided.")
            if mean is None:
                mean = torch.zeros(d, dtype=X.dtype, device=self.device)
            else:
                mean = mean.to(self.device, dtype=X.dtype)
            covariance_matrix = covariance_matrix.to(self.device, dtype=X.dtype)

            self.variational(
                mean=mean,
                covariance_matrix=covariance_matrix,
                epochs=epochs,
                lr=lr
            )
    
    def predict_proba_laplace(self, X):
        X = X.to(self.device, dtype=self.theta.dtype)
        probs = []

        for x in X:
            mu = x @ self.theta
            sigma2 = x @ self.cov @ x

            kappa = torch.sqrt(1 + (torch.pi / 8) * sigma2)
            prob = torch.sigmoid(mu / kappa)

            probs.append(prob)

        return torch.stack(probs)
    
    def predict_proba_variational(self, X, M=50):
        X = X.to(self.device, dtype=self.theta.dtype)
        probs = []

        for x in X:
            samples = []

            for _ in range(M):
                eps = torch.randn_like(self.theta)
                theta_sample = self.theta + self.sigma * eps
                prob = torch.sigmoid(x @ theta_sample)
                samples.append(prob)

            probs.append(torch.mean(torch.stack(samples)))

        return torch.stack(probs)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device, dtype=self.theta.dtype)
        if self.mode == 'laplace':
            probabilities = self.predict_proba_laplace(X)
        elif self.mode == 'maximum_posteriori':
            probabilities = self.sigmoid(X @ self.theta)
        elif self.mode == 'variational':
            probabilities = self.predict_proba_variational(X)
        return (probabilities >= 0.5).float()
