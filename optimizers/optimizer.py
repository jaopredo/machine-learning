from abc import ABC, abstractmethod
from core import Parameter


class Optimizer(ABC):
    def __init__(self, parameters: list[Parameter]) -> None:
        self.parameters: list[Parameter] = list(parameters)
    
    @abstractmethod
    def step(self) -> None:
        """Update the parameters based on their gradients."""
        pass

    def zero_grad(self) -> None:
        """Reset the gradients of all parameters to zero."""
        for param in self.parameters:
            param.zero_grad()
