from __future__ import annotations

from abc import ABC
from collections.abc import Iterable

import torch


class Module(ABC):
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self) -> list[torch.nn.Parameter]:
        return []

    def to(self, device: torch.device | str):
        return self

    def zero_grad(self):
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()

    @staticmethod
    def _move_tensor(value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return value

    @staticmethod
    def _move_nested(values: Iterable, device: torch.device):
        return [Module._move_tensor(value, device) for value in values]
