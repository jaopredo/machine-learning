import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):
    def forward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8  # para evitar log(0)
        Y = torch.clamp(Y, epsilon, 1 - epsilon)  # limita os valores entre epsilon e 1-epsilon
        loss = - (T * torch.log(Y) + (1 - T) * torch.log(1 - Y))
        return loss.mean()
    
    def backward(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8  # para evitar divisão por zero
        Y = torch.clamp(Y, epsilon, 1 - epsilon)  # limita os valores entre epsilon e 1-epsilon
        dY = (Y - T) / (Y * (1 - Y))  # derivada da BCE em relação a Y
        return dY / Y.shape[0]  # média sobre o batch


class BinaryCrossEntropyWithLogitsLoss(nn.Module):
    def forward(self, logits: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        # Stable BCE with logits: max(0, x) - x*T + log(1 + exp(-|x|))
        loss = torch.clamp(logits, min=0) - logits * T + torch.log1p(torch.exp(-torch.abs(logits)))
        return loss.mean()

    def backward(self, logits: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        sigmoid = torch.sigmoid(logits)
        return (sigmoid - T) / logits.shape[0]
