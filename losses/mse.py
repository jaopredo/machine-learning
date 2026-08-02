import torch
from .loss import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        self._prediction: torch.Tensor | None = None
        self._target: torch.Tensor | None = None

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if y.shape != t.shape:
            raise ValueError("Prediction and target must have the same shape")
        self._prediction = y
        self._target = t
        return torch.mean((y - t) ** 2)

    def backward(self) -> torch.Tensor:
        if self._prediction is None or self._target is None:
            raise ValueError("Must call forward() before backward()")
        N = self._prediction.numel()
        return (2/N) * (self._prediction - self._target)

    @property
    def prediction(self) -> torch.Tensor:
        if self._prediction is None or self._target is None:
            raise ValueError("Must call forward() before backward()")
        return self._prediction

    @property
    def target(self) -> torch.Tensor:
        if self._prediction is None or self._target is None:
            raise ValueError("Must call forward() before backward()")
        return self._target

