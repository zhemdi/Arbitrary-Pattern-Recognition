import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .pooling import SO3Softmax

EPS = 1e-16
NORM_FACTOR = 3.0



class SO3GlobalActivation(nn.Module):
    """
    Global activation module for SO(3) equivariant neural networks.

    Args:
        order (int): The maximum order of the SO(3) representation.
        num_feat (int): The number of features. Default is 4.
        activation (str): The activation function to use.
            Choices: 'sigmoid', 'relu', 'silu'. Default is 'sigmoid'.
    """
    def __init__(self, order, num_feat=4, activation='sigmoid'):
        super(SO3GlobalActivation, self).__init__()
        self.order = order
        self.num_feat = num_feat
        self.W = nn.Parameter(torch.rand(self.num_feat, self.num_feat))
        self.b = nn.Parameter(torch.rand(self.num_feat))
        self.SO3Softmax = SO3Softmax(order, num_feat=num_feat)

        def activation_fun(x):
            if activation == 'sigmoid':
                return torch.nn.functional.sigmoid(x)
            elif activation == 'relu':
                return torch.nn.functional.relu(x)
            elif activation == 'silu':
                return torch.nn.functional.silu(x)

        self.activation_fun = activation_fun

    def forward(self, input):
        """
        Forward pass of the SO3GlobalActivation module.

        Args:
            input (torch.Tensor): Input tensor of shape
                (batch_size, num_feat, order, height, width, depth).

        Returns:
            torch.Tensor: Output tensor after applying global activation.
        """
        fun_softmax = self.SO3Softmax(input)[:, :, None]
        fun_norm_ret = self.activation_fun(
            torch.einsum('bdixyz,ed->beixyz', fun_softmax, self.W)
            + self.b[None, :, None, None, None, None]
        )
        out = input * fun_norm_ret
        return out