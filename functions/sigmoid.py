import torch
import torch.nn as nn


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = None  # cache para a passagem backward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = 1 / (1 + torch.exp(-x))
        return self.out

    def backward(self, d_output: torch.Tensor) -> torch.Tensor:
        sigmoid_derivative = self.out * (1 - self.out)
        return d_output * sigmoid_derivative
