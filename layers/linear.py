import torch
from core import Parameter, Module


class LinearLayer(Module):
    def __init__(self, input_dim: int, output_dim: int, device: torch.device = torch.device("cpu")):
        self.__device: torch.device = device

        self.W: Parameter = Parameter(torch.randn(input_dim, output_dim, device=self.__device) * 0.01)
        self.b: Parameter = Parameter(torch.zeros(output_dim, device=self.__device))

        self.x: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        if dout.device != self.__device:
            raise TypeError(f"Gradient tensor is on device {dout.device}, but layer is on device {self.__device}.")
        if self.x is None:
            raise ValueError("No forward pass has been called before backward.")
        
        self.W.accumulate_grad(self.x.T @ dout)
        self.b.accumulate_grad(torch.sum(dout, dim=0))
        dx = dout @ self.W.T

        return dx

    def to(self, device: torch.device):
        self.__device = device

        self.W.to(device)
        self.b.to(device)

        self.x = self.x.to(device) if self.x is not None else None

        return self
