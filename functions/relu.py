import torch
import torch.nn as nn

class LeakyReLU(nn.Module):
    def __init__(self, alpha=0.01):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.x = None  # cache

    def forward(self, x):
        self.x = x
        return torch.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        dx = torch.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return dout * dx