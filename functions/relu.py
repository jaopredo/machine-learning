import torch
from core import Module


class ReLU(Module):
    def __init__(self):
        self.mask: torch.Tensor | None = None  # cache for backward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            raise ValueError("Forward pass must be called before backward pass.")
        return dout * self.mask

    def to(self, device: torch.device):
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self
