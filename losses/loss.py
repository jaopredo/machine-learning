from abc import ABC, abstractmethod
import torch


class Loss(ABC):
    @abstractmethod
    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def backward(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def prediction(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def target(self) -> torch.Tensor:
        pass
