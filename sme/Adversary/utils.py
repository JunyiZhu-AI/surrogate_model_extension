import torch

__all__ = [
    "require_grad",
    "prior_boundary",
    "compute_norm",
    "total_variation"
]


def require_grad(net, flag):
    for p in net.parameters():
        p.require_grad = flag


def prior_boundary(data, low, high):
    with torch.no_grad():
        data.data = torch.clamp(data, low, high)


def compute_norm(inputs):
    squared_sum = sum([p.square().sum() for p in inputs])
    norm = squared_sum.sqrt()
    return norm


def total_variation(x):
    dh = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()
    dw = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    return (dh + dw) / 2
