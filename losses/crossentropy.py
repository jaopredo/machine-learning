import torch
from .loss import Loss


class CrossEntropy(Loss):
    def __init__(self):
        self._prediction: torch.Tensor | None = None
        self._target: torch.Tensor | None = None

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if y.shape != t.shape:
            raise ValueError(f"Shape mismatch: y has shape {y.shape}, but t has shape {t.shape}.")
        if y.dim() != 2 or t.dim() != 2:
            raise ValueError(f"Expected 2D tensors for y and t, but got y with shape {y.shape} and t with shape {t.shape}.")

        self._prediction = y
        self._target = t
        
        # Compute the cross-entropy loss
        loss = -torch.sum(self._target * torch.log(self._prediction + 1e-9)) / y.size(0)

        return loss

    def backward(self) -> torch.Tensor:
        if self._prediction is None or self._target is None:
            raise ValueError("No predictions or targets available for backpropagation. You must call forward() before backward().")

        # Compute the gradient of the loss with respect to the predictions (probabilities)
        grad = - (self._target / (self._prediction + 1e-9)) / self._prediction.size(0)

        return grad

    @property
    def prediction(self) -> torch.Tensor | None:
        return self._prediction

    @property
    def target(self) -> torch.Tensor | None:
        return self._target
