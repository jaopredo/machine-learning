import torch
from abc import ABC, abstractmethod
from .parameter import Parameter


class Module(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "Module":
        pass

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                params.append(attr)

        return params
