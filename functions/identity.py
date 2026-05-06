import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def backward(self, d_output: torch.Tensor) -> torch.Tensor:
        return d_output
