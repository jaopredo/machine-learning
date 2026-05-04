import torch
import torch.nn as nn


def im2col(x, kernel_size, stride=1, padding=0):
    N, C, H, W = x.shape
    F = kernel_size

    x_padded = torch.nn.functional.pad(
        x,
        (padding, padding, padding, padding)
    )

    H_out = (H + 2 * padding - F) // stride + 1
    W_out = (W + 2 * padding - F) // stride + 1

    cols = []

    for i in range(F):
        for j in range(F):
            patch = x_padded[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride]
            cols.append(patch)

    cols = torch.stack(cols)                        # (F*F, N, C, H_out, W_out)
    cols = cols.permute(1, 2, 0, 3, 4)             # (N, C, F*F, H_out, W_out)
    cols = cols.reshape(N, C * F * F, H_out * W_out)

    return cols


def col2im(cols, x_shape, kernel_size, stride=1, padding=0):
    N, C, H, W = x_shape
    F = kernel_size

    H_out = (H + 2 * padding - F) // stride + 1
    W_out = (W + 2 * padding - F) // stride + 1

    x_padded = torch.zeros(
        (N, C, H + 2 * padding, W + 2 * padding),
        device=cols.device
    )

    cols = cols.reshape(N, C, F * F, H_out, W_out)

    idx = 0
    for i in range(F):
        for j in range(F):
            x_padded[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride] += cols[:, :, idx]
            idx += 1

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded