import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .activation import SO3LocalActivation

EPS = 1e-16


class AvgPoolSE3(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        """Initialize the AvgPoolSE3 module.

        Args:
            kernel_size: The size of the window to take an average over.
            stride: The stride of the window. Default value is kernel_size.
            padding: Implicit zero padding to be added on both sides.
        """
        super(AvgPoolSE3, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        """Forward pass for the AvgPoolSE3 module.

        Args:
            x: List of input tensors with shapes [(B, N, 2l+1, 2l+1, H, W, D) for l in range(L)].

        Returns:
            Output with average pooling applied.
        """
        avg = F.avg_pool3d(torch.reshape(x, [x.size(0), -1, x.size(3), x.size(4), x.size(5)]),
                           self.kernel_size, self.stride, self.padding)
        
        return torch.reshape(avg, [x.size(0), x.size(1), x.size(2), avg.size(2), avg.size(3), avg.size(4)])


class SO3Softmax(nn.Module):
    def __init__(self, order, distr_dependency=False, num_feat=4, global_activation=False, coefficients_type='trainable'):
        super(SO3Softmax, self).__init__()
        self.coefficients_type = coefficients_type
        self.so3_trainable_activation = SO3LocalActivation(order, distr_dependency, num_feat, coefficients_type =  coefficients_type)
        self.order_out = 2 * order - 1
        self.order = order
        self.global_activation = global_activation

    def forward(self, input):
        output = self.so3_trainable_activation(input)
        degree_mult = torch.cat([torch.ones([(2 * l + 1) ** 2], device=input.device) * (2 * l + 1) for l in range(self.order_out)], dim=0)
        so3_softmax_output = torch.sum(output ** 2 / degree_mult[None, None, :, None, None, None], dim=2)
        so3_softmax_output /= torch.abs(output[:, :, 0] + EPS)
        return so3_softmax_output