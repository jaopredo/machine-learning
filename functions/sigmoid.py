import torch
from core import Module


class Sigmoid(Module):
    def __init__(self):
        self.out: torch.Tensor | None = None  # cache the output for backward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = 1 / (1 + torch.exp(-x))
        return self.out

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        if self.out is None:
            raise ValueError("Forward pass must be called before backward pass.")
        sigmoid_derivative = self.out * (1 - self.out)
        return dout * sigmoid_derivative

    def to(self, device: torch.device):
        if self.out is not None:
            self.out = self.out.to(device)
        return self
