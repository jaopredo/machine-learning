import torch
import torch.nn as nn


class MeanSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.mean((Y - T) ** 2)

    def backward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return 2 * (Y - T) / Y.shape[0]  # média sobre o batch