import torch
from math import sqrt

from core import Module, Parameter


class GCNLayer(Module):
    def __init__(self, input_dim: int, output_dim: int, norm_adjacency: torch.Tensor, device: torch.device = torch.device("cpu")):
        """Linear layer for Graph Convolutional Networks

        Args:
            dimensions (tuple[int, int]): Tuple containing the number of input and output features
            activation (nn.Module): Activation function for the layer
        """
        super().__init__()

        self.__device: torch.device = device

        # Inicializando os pesos
        # Pesos
        self.W: Parameter = Parameter(torch.randn(input_dim, output_dim, device=self.__device) * sqrt(2 / output_dim))
        # Bias
        self.b: Parameter = Parameter(torch.zeros(output_dim, device=device))

        # Caches
        self.messages: torch.Tensor|None = None  # cache for backward pass
        self.previous_embeddings: torch.Tensor|None  = None  # cache for backward pass
        self.norm_adjacency: torch.Tensor = norm_adjacency  # cache for backward pass

    def to(self, device: torch.device):
        self.__device = device

        self.W.to(device)
        self.b.to(device)
        self.norm_adjacency = self.norm_adjacency.to(device)

        self.messages = self.messages.to(device) if self.messages is not None else None

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the linear layer

        Args:
            previous_embeddings (torch.Tensor): Embeddings generated in the previous layer

        Returns:
            torch.Tensor: Output graph features, shape (n_nodes, out_features)
        """
        previous_embeddings = x

        one = torch.ones(previous_embeddings.shape[0], 1, device=self.__device, dtype=previous_embeddings.dtype)
        self.previous_embeddings = previous_embeddings  # Cache the input for use in the backward pass

        self.messages = self.norm_adjacency @ previous_embeddings @ self.W + one @ self.b.unsqueeze(0)  # shape (n_nodes, out_features)
        assert self.messages is not None

        return self.messages

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        assert self.previous_embeddings is not None

        self.W.accumulate_grad((self.norm_adjacency @ self.previous_embeddings).t() @ dout)
        self.b.accumulate_grad(dout.sum(dim=0))

        dx = self.norm_adjacency.t() @ dout @ self.W.t()

        return dx
