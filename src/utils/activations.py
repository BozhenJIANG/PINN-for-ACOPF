"""
Custom activation functions for neural networks.
"""
import torch
import torch.nn as nn


class BetaSiLU(nn.Module):
    """
    Beta-scaled Sigmoid Linear Unit activation function.
    
    Args:
        beta: Scaling factor for the sigmoid input
    """
    def __init__(self, beta=1.0):
        super(BetaSiLU, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class MinMaxSigmoid(nn.Module):
    """
    Sigmoid activation scaled to [0.9, 1.1] range.
    Useful for voltage magnitude constraints.
    """
    def forward(self, x):
        return 0.9 + 0.2 * torch.sigmoid(x)


def beta_silu(x, beta=1.0):
    """Functional version of BetaSiLU."""
    return x * torch.sigmoid(beta * x)


def min_max_sigmoid(x):
    """Functional version of MinMaxSigmoid."""
    return 0.9 + 0.2 * torch.sigmoid(x)
