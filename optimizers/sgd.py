from __future__ import annotations

import torch


class SGD:
    def __init__(self, parameters, lr: float = 0.01, max_norm: float | None = None):
        self.parameters = list(parameters)
        self.lr = lr
        self.max_norm = max_norm

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad
                if self.max_norm is not None:
                    norm = torch.norm(gradient)
                    if norm > self.max_norm:
                        gradient = gradient / norm * self.max_norm

                parameter -= self.lr * gradient

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()
