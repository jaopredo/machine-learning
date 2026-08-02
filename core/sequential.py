import torch
from .parameter import Parameter
from .module import Module


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules: list[Module] = list(modules)
        if len(self.modules) == 0:
            raise ValueError("Sequential must contain at least one module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        for module in reversed(self.modules):
            dout = module.backward(dout)
        return dout

    def to(self, device: torch.device):
        for module in self.modules:
            module.to(device)
        return self

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
