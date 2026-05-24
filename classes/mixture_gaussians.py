from typing import Callable
from numpy.ma import indices
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class MixtureGaussians(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        convergence_criterium: Callable[[torch.Tensor, torch.Tensor], bool],
        max_iterations: int = 1000,
        device=torch.device('cpu')
    ):
        """Mixture of Gaussians model

        Args:
            k (int): Number of Gaussian components in the mixture.
            d (int): Dimensionality of the data.
            convergence_criterium (Callable[[torch.Tensor, torch.Tensor], bool]): Function that
            receives 2 tensors as parameters, if the function returns True, is because the two tensors are
            close enough to consider that they converged. If it returns False, they are not close enough
            to consider to be converged
        """
        super().__init__()

        self.k = k
        self.d = d
        self.device = device
        self.convergence_criterium = convergence_criterium
        self.max_iterations = max_iterations


        # Initializing the parameters of the mixture of Gaussians

        # Means of the Gaussians
        self.mu = torch.randn(k, d, device=self.device)
        # Precision matrices (inverse of covariance) of the Gaussians
        self.covariances = torch.eye(d, device=self.device).unsqueeze(0).repeat(k, 1, 1)  # (K, D, D)
        # Adding a small value to the diagonal for numerical stability
        
        # Mixing coefficients (weights) for each Gaussian component
        self.pi = torch.ones(k, device=self.device) / k

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood of the data under the mixture of Gaussians model.

        Args:
            x (torch.Tensor): Input data of shape (n_samples, d).
        Returns:
            torch.Tensor: Log-likelihood of the data under the model.
        """
        N, _ = X.shape

        indices = torch.randperm(N)[:self.k]
        self.mu = X[indices].clone()  # (K, D)
        
        if self.d != X.shape[1]:
            raise ValueError(f"Input data must have dimensionality {self.d}, but got {X.shape[1]}")
        
        # Criteriums for MEANS, COVARIANCES and PROBABILITIES
        previous = [
            self.mu.clone(),
            self.covariances.clone(),
            self.pi.clone()
        ]
        criteriums = [
            False,
            False,
            False
        ]

        for _ in range(self.max_iterations):
            # Step E
            log_probs = []
            for k in range(self.k):
                dist = MultivariateNormal(loc=self.mu[k], covariance_matrix=self.covariances[k])
                log_probs.append(dist.log_prob(X))  # (N,)

            log_probs = torch.stack(log_probs, dim=1)  # (N, K)

            log_pi = torch.log(self.pi)
            log_weighted = log_pi + log_probs

            log_denom = torch.logsumexp(log_weighted, dim=1, keepdim=True)  # (N, 1)
            log_r = log_weighted - log_denom  # (N, K)

            r = log_r.exp()  # (N, K)

            # Step M
            Ns = r.sum(dim=0)  # (K,)
            
            self.mu = (r.T @ X) / Ns.unsqueeze(1)

            eps = 1e-6

            for k in range(self.k):
                diff = X - self.mu[k]                         # (N, D)
                weighted = r[:, k].unsqueeze(1) * diff        # (N, D)
                self.covariances[k] = (weighted.T @ diff) / Ns[k] + eps * torch.eye(self.d, device=self.device)
            
            self.pi = Ns / N

            # Criterium step
            criteriums[0] = self.convergence_criterium(self.mu, previous[0])
            criteriums[1] = self.convergence_criterium(self.covariances, previous[1])
            criteriums[2] = self.convergence_criterium(self.pi, previous[2])

            if sum(criteriums) == 3:
                break
    
    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood of the data under the mixture of Gaussians model.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, d).
        Returns:
            torch.Tensor: Log-likelihood scalar.
        """
        log_probs = []
        for k in range(self.k):
            dist = MultivariateNormal(loc=self.mu[k], covariance_matrix=self.covariances[k])
            log_probs.append(dist.log_prob(X))  # (N,)

        log_probs = torch.stack(log_probs, dim=1)  # (N, K)

        log_pi = torch.log(self.pi)
        log_weighted = log_pi + log_probs  # (N, K)

        # log p(x_n) = log sum_k pi_k * N_k  →  logsumexp over K
        log_px = torch.logsumexp(log_weighted, dim=1)  # (N,)

        # soma sobre todos os pontos
        return log_px.sum()
