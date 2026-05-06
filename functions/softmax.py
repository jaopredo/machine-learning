import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.out = None  # cache da saída

    def forward(self, x):
        """
        x: (n_classes, batch_size) ou (n_classes,) se 1 amostra
        """
        # estabilidade numérica
        x_shifted = x - torch.max(x, dim=0, keepdim=True).values
        
        exp_x = torch.exp(x_shifted)
        out = exp_x / torch.sum(exp_x, dim=0, keepdim=True)

        self.out = out
        return out

    def backward(self, dout):
        """
        dout: dL/dy (mesmo shape da saída)
        retorna: dL/dx
        """
        dx = torch.zeros_like(self.out)

        # caso batch
        if self.out.ndim == 2:
            for i in range(self.out.shape[1]):  # loop no batch
                y = self.out[:, i].reshape(-1, 1)  # coluna
                
                # Jacobiano da softmax
                J = torch.diagflat(y) - y @ y.T
                
                dx[:, i] = (J @ dout[:, i]).reshape(-1)
        else:
            y = self.out.reshape(-1, 1)
            J = torch.diagflat(y) - y @ y.T
            dx = (J @ dout.reshape(-1, 1)).reshape(-1)

        return dx