import torch
import torch.nn as nn
import numpy as np
from utils.imcol import im2col, col2im


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()

        self.padding = padding
        self.stride = stride

        std = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        self.b = nn.Parameter(torch.zeros(out_channels))

        self.dW = None
        self.db = None

        self.cache = {}

    def forward(self, x):
        N, C, H, W = x.shape
        F = self.W.shape[-1]

        X_col = im2col(x, F, self.stride, self.padding)    # (N, C*F^2, H_out*W_out)
        W_col = self.W.view(self.W.shape[0], -1)            # (kappa, C*F^2)

        H_out = (H + 2 * self.padding - F) // self.stride + 1
        W_out = H_out

        # (kappa, C*F^2) @ (N, C*F^2, H_out*W_out) -> precisa de broadcast
        # reshape X_col pra (N, C*F^2, H_out*W_out) -> transpõe pra (N, H_out*W_out, C*F^2)
        # produto: (N, kappa, H_out*W_out)
        out = torch.einsum('kc,ncp->nkp', W_col, X_col)    # (N, kappa, H_out*W_out)
        out = out + self.b.view(1, -1, 1)                   # bias broadcast
        out = out.reshape(N, self.W.shape[0], H_out, W_out)

        self.cache = {
            "x_shape": x.shape,
            "X_col": X_col,
            "W_col": W_col
        }

        return out

    def backward(self, dO):
        X_col = self.cache["X_col"]     # (N, C*F^2, H_out*W_out)
        W_col = self.cache["W_col"]     # (kappa, C*F^2)
        x_shape = self.cache["x_shape"]

        N, kappa, H_out, W_out = dO.shape
        dO_flat = dO.reshape(N, kappa, H_out * W_out)       # (N, kappa, H_out*W_out)

        # grad W: soma sobre N e posições espaciais
        # dW[k, c] = sum_n sum_p dO_flat[n,k,p] * X_col[n,c,p]
        self.dW = torch.einsum('nkp,ncp->kc', dO_flat, X_col).view_as(self.W)

        # grad b: soma sobre N e posições espaciais
        self.db = dO_flat.sum(dim=(0, 2))                   # (kappa,)

        # grad input
        # dX_col[n,c,p] = sum_k W_col[k,c] * dO_flat[n,k,p]
        dX_col = torch.einsum('kc,nkp->ncp', W_col, dO_flat)  # (N, C*F^2, H_out*W_out)
        dX = col2im(dX_col, x_shape, self.W.shape[-1], self.stride, self.padding)

        return dX


class MaxPoolLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.cache = {}

    def forward(self, x):
        N, C, H, W = x.shape
        F = self.kernel_size

        H_out = (H - F) // self.stride + 1
        W_out = (W - F) // self.stride + 1

        x_col = im2col(x, F, self.stride, padding=0)        # (N, C*F^2, H_out*W_out)
        x_col = x_col.reshape(N, C, F * F, H_out * W_out)  # (N, C, F^2, H_out*W_out)

        out, max_idx = x_col.max(dim=2)                     # (N, C, H_out*W_out)
        out = out.reshape(N, C, H_out, W_out)

        self.cache = {
            "x_shape": x.shape,
            "x_col": x_col,
            "max_idx": max_idx
        }

        return out

    def backward(self, dO):
        N, C, H, W = self.cache["x_shape"]
        x_col = self.cache["x_col"]         # (N, C, F^2, H_out*W_out)
        max_idx = self.cache["max_idx"]     # (N, C, H_out*W_out)
        F = self.kernel_size

        H_out = (H - F) // self.stride + 1
        W_out = (W - F) // self.stride + 1

        dO_flat = dO.reshape(N, C, H_out * W_out)           # (N, C, H_out*W_out)

        # scatter gradiente só pras posições que eram máximo
        dX_col = torch.zeros_like(x_col)                    # (N, C, F^2, H_out*W_out)
        dX_col.scatter_(2, max_idx.unsqueeze(2), dO_flat.unsqueeze(2))

        dX_col = dX_col.reshape(N, C * F * F, H_out * W_out)
        dX = col2im(dX_col, self.cache["x_shape"], F, self.stride, padding=0)

        return dX


class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha
        self.cache = {
            'input': None,
        }

    def forward(self, x):
        self.cache['input'] = x
        return torch.where(x > 0, x, self.alpha * x)

    def backward(self, dO):
        dX = torch.where(self.cache['input'] > 0, dO, self.alpha * dO)
        return dX


class FullConectedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        std = np.sqrt(2 / in_features)
        self.W = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.b = nn.Parameter(torch.zeros(out_features))

        self.cache = {
            "input": None,
            "pre_activation": None,
            "pool_mask": None,
            "dropout_mask": None
        }
    
    def forward(self, x):
        self.cache["input"] = x
        return x @ self.W.T + self.b
    
    def backward(self, dO):
        x = self.cache["input"]
        self.dW = dO.T @ x
        self.db = dO.sum(dim=0)
        dX = dO @ self.W
        return dX


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = {
            "input_shape": None
        }

    def forward(self, x):
        self.cache["input_shape"] = x.shape
        return x.view(x.size(0), -1)

    def backward(self, dO):
        return dO.view(self.cache["input_shape"])


class CNN(nn.Module):
    def __init__(self,
        conv_dimensions: list[tuple[int, int, int]] = [(3, 6, 5), (6, 16, 5)],
        connected_dimensions: list[int] = [120, 84, 10],
        pooling_params: list[tuple[int, int]] = [(2, 2), (2, 2)],
        paddings: list[int] = [2, 0],
        strides: list[int] = [1, 1],
        relu_parameters: list[float] = None,
    ):
        super().__init__()

        if relu_parameters is None:
            relu_parameters = [0.0] * (len(conv_dimensions) + len(connected_dimensions) - 2)

        assert len(relu_parameters) == len(conv_dimensions) + len(connected_dimensions) - 2, "Deve haver um valor de parâmetro de ReLU para cada camada, exceto a última FC."
        assert len(pooling_params) == len(conv_dimensions), "Deve haver um conjunto de parâmetros de pooling para cada camada convolucional, exceto a última."
        assert len(paddings) == len(conv_dimensions), "Deve haver um valor de padding para cada camada convolucional."
        assert len(strides) == len(conv_dimensions), "Deve haver um valor de stride para cada camada convolucional."

        self.layers = nn.ModuleList()
        for i in range(len(conv_dimensions)):
            # Camadas convolucionais + ReLU
            self.layers.append(
                ConvolutionalLayer(
                    conv_dimensions[i][0],
                    conv_dimensions[i][1],
                    conv_dimensions[i][2],
                    paddings[i],
                    strides[i]
                )
            )
            self.layers.append(
                LeakyReLU(alpha=relu_parameters[i])
            )

            # Pool (se existir)
            if i < len(pooling_params):
                self.layers.append(
                    MaxPoolLayer(
                        kernel_size=pooling_params[i][0],
                        stride=pooling_params[i][1]
                    )
                )
        
        # Flatten
        self.layers.append(FlattenLayer())

        # descobre o tamanho da saída do flatten automaticamente
        with torch.no_grad():
            dummy = torch.zeros(1, conv_dimensions[0][0], 32, 32)
            for layer in self.layers:
                dummy = layer.forward(dummy)
            in_features = dummy.shape[1]

        # Camadas totalmente conectadas + ReLU
        dims = [in_features] + connected_dimensions
        for i in range(len(connected_dimensions)):
            self.layers.append(
                FullConectedLayer(dims[i], dims[i+1])
            )
            if i < len(connected_dimensions) - 1:
                self.layers.append(
                    LeakyReLU(alpha=relu_parameters[len(conv_dimensions) + i - 2])
                )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dO):
        for layer in reversed(self.layers):
            dO = layer.backward(dO)
        return dO