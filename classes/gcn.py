import numpy as np
import networkx as nx
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, dimensions: tuple[int,int], activation: nn.Module, device: torch.device|str|None=None):
        """Linear layer for Graph Convolutional Networks

        Args:
            dimensions (tuple[int, int]): Tuple containing the number of input and output features
            activation (nn.Module): Activation function for the layer
        """
        super().__init__()

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.W = nn.Parameter(torch.randn(dimensions[0], dimensions[1], device=device) * np.sqrt(2 / dimensions[1]))
        self.activation = activation
        self.b = nn.Parameter(torch.zeros(dimensions[1], device=device))
        self.H = None  # cache for backward pass
        self.Z = None  # cache for backward pass
        self.prev_Z = None  # cache for backward pass
        self.A = None  # cache for backward pass
        self.D = None  # cache for backward pass

    def set_adjacency_matrix(self, A: torch.Tensor):
        """Sets the adjacency matrix for the current layer. This method should be called before the forward pass of the layer.

        Args:
            A (torch.Tensor): Adjacency matrix of the graph, shape (n_nodes, n_nodes)
        """
        self.A = A

    def forward(self, prev_Z: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the linear layer

        Args:
            prev_Z (torch.Tensor): Previous layer's output, shape (n_nodes, in_features)
            A (torch.Tensor): Adjacency matrix of the graph, shape (n_nodes, n_nodes)

        Returns:
            torch.Tensor: Output graph features, shape (n_nodes, out_features)
        """
        one = torch.ones(prev_Z.shape[0], 1, device=prev_Z.device, dtype=prev_Z.dtype)
        self.prev_Z = prev_Z  # Cache the input for use in the backward pass

        self.H = self.A @ prev_Z @ self.W + one @ self.b.unsqueeze(0)  # shape (n_nodes, out_features)
        self.Z = self.activation.forward(self.H)
        return self.Z

    def backward(self, dZ: torch.Tensor, penalty: float = 0.001) -> torch.Tensor:
        self.D = self.activation.backward(dZ)
        self.W.grad =  (self.A @ self.prev_Z).t() @ self.D + penalty * self.W
        self.b.grad = self.D.sum(dim=0)

        dZ_prev = self.A.t() @ self.D @ self.W.t()
        return dZ_prev



class GraphConvolutionNetwork(nn.Module):
    def __init__(
        self,
        G: nx.DiGraph,
        layers_dimensions: list[int],
        output_activation: nn.Module,
        activations: list[nn.Module],
        loss_func: nn.Module,
        device: torch.device | str | None = None
    ):
        """Initializes a MultilayerPerceptron that can be used for regressions and classifications depending on the
        `activation` functions passed. 
    
        Parameters
        ----------
        layers_dimensions: list[int]
            The list containing the amount of features per layer (Do not include the BIAS)
        activations : list[Callable[[np.ndarray], np.ndarray]]
            List containing the activation functions used in the hidden layers, where the i-th function of
            the list is assigned to the i-th layer
        activations_derivatives: list[Callable[[np.ndarray], np.ndarray]]
            List containing the derivatives of each function passed in the `activations` argument. If
            `activations` has `n` entries, this argument must have `n-1` as we don't use the derivative
            of the last layer (output), for that, the target column must be adapted
        error_function: nn.Module
            The error function used to calculate the error history on the training method
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.G = G  # cache for backward pass

        self.X = torch.stack([G.nodes[i]['x'] for i in G.nodes], dim=0).to(self.device)  # shape (n_nodes, in_features)

        # Getting a dense adjacency matrix from the input graph and converting it to a PyTorch tensor
        self.A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=torch.float32, device=self.device)  # shape (n_nodes, n_nodes)
        self.A += torch.eye(self.A.shape[0], device=self.device)  # Adding self-loops to the adjacency matrix

        # Computing the degrees vector of the graph to normalize the adjacency matrix
        d = self.A.sum(dim=1)  # shape (n_nodes,)
        d_inv_sqrt = torch.pow(d, -0.5)

        D = torch.diag(d_inv_sqrt)  # shape (n_nodes, n_nodes)
        self.norm_A = D @ self.A @ D  # Normalized adjacency matrix, shape (n_nodes, n_nodes)

        self.loss_func: nn.Module = loss_func

        for i in range(len(layers_dimensions) - 1):
            layer = None
            if i == len(layers_dimensions) - 2:  # última camada
                layer = LinearLayer(
                    (layers_dimensions[i], layers_dimensions[i + 1]),
                    output_activation,
                    self.device
                )
            else:
                layer = LinearLayer(
                    (layers_dimensions[i], layers_dimensions[i + 1]),
                    activations[i],
                    self.device
                )
            if isinstance(layer, LinearLayer):
                layer.set_adjacency_matrix(self.norm_A)  # Set the normalized adjacency matrix for the current layer
            self.layers.append(layer)


    def forward(self):
        """Performs the forward pass of the MLP

        Args:
            G (nx.DiGraph): Input graph with node features stored in the 'x' attribute of each node

        Returns:
            torch.Tensor: Output of the network, shape (n_samples, output_features)
        """
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, Y: torch.Tensor, T: torch.Tensor, mask: torch.Tensor, penalty: float = 0.001):
        # Loss
        Y = Y.squeeze()  # shape (n_nodes,)
        loss = self.loss_func(Y[mask], T[mask])
        
        # Gradient Loss
        dZ = torch.zeros_like(Y)
        dZ[mask] = self.loss_func.backward(Y[mask], T[mask])

        if dZ.dim() == 1:
            dZ = dZ.unsqueeze(1)

        # Backpropagation through layers
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ, penalty)
    
    def update(self, learning_rate: float, max_norm: float = 5.0):
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm > max_norm:
                        param.grad = param.grad / norm * max_norm
                    param -= learning_rate * param.grad
                    param.grad.zero_()
