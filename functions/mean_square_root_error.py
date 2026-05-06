import torch
import torch.nn as nn


class MeanSquareRootError(nn.Module):
    def forward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(torch.mean((Y - T) ** 2))
        return loss
    
    def backward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        dY = (Y - T) / (torch.sqrt(torch.mean((Y - T) ** 2)) + 1e-8)
        return dY / Y.shape[0]  # média sobre o batch
