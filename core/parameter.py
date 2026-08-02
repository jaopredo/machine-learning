from __future__ import annotations
import torch


class Parameter:
    def __init__(self, data: torch.Tensor):
        self.__data: torch.Tensor = data
        self.__grad: torch.Tensor | None = None

    def to(self, device: torch.device):
        self.__data = self.__data.to(device)
        if self.__grad is not None:
            self.__grad = self.__grad.to(device)
        return self

    def accumulate_grad(self, grad: torch.Tensor):
        if self.__grad is None:
            self.__grad = grad
        else:
            self.__grad += grad

    def zero_grad(self):
        self.__grad = None

    @property
    def data(self) -> torch.Tensor:
        return self.__data

    @data.setter
    def data(self, new_data: torch.Tensor):
        if new_data.shape != self.__data.shape:
            raise ValueError(f"New data must have the same shape as the original data. Expected {self.__data.shape}, got {new_data.shape}.")
        self.__data = new_data

    @property
    def grad(self) -> torch.Tensor | None:
        return self.__grad

    @grad.setter
    def grad(self, _: torch.Tensor | None):
        raise AttributeError("Direct assignment to 'grad' is not allowed. Use 'accumulate_grad' or 'zero_grad' methods instead.")

    def __getattr__(self, name: str):
        # Redireciona atributos e métodos como .shape, .device, .to(), .t(), etc.
        return getattr(self.__data, name)

    # Operadores matriciais
    def __matmul__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return self.__data @ other_data

    def __rmatmul__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return other_data @ self.__data

    # Operadores aritméticos básicos
    def __add__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return self.__data + other_data

    def __radd__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return other_data + self.__data

    def __sub__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return self.__data - other_data

    def __rsub__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return other_data - self.__data

    def __mul__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return self.__data * other_data

    def __rmul__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return other_data * self.__data

    def __truediv__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return self.__data / other_data

    def __rtruediv__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        return other_data / self.__data

    def __neg__(self):
        return -self.__data

    # Operadores in-place (essenciais para o SGD, ex: parameter -= lr * grad)
    def __isub__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        self.__data -= other_data
        return self

    def __iadd__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        self.__data += other_data
        return self

    def __imul__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        self.__data *= other_data
        return self

    def __itruediv__(self, other: torch.Tensor | Parameter):
        other_data = other.__data if isinstance(other, Parameter) else other
        self.__data /= other_data
        return self

    # Representação amigável do parâmetro
    def __repr__(self):
        return f"Parameter(\n{self.__data}\n)"
