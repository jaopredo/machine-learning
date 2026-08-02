from .optimizer import Optimizer
from core import Parameter


class SGD(Optimizer):
    def __init__(self, parameters: list[Parameter], learning_rate: float = 0.01):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad
