import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basis_functions import spherical_harmonics
from .compute_filters import ComputeFiltersFunction
from .cg_coefficients import clebsch_gordan


class EquiLocalPatOrientConvolution(nn.Module):
    """
    Equivariant Local Patch Orientation Convolution module.

    This module performs convolution on 3D data while preserving equivariance
    to rotations and translations.

    Args:
        num_inputs (int): Number of input channels.
        num_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        order_in (int): Order of the input spherical harmonics. Default is 1.
        order_out (int): Order of the output spherical harmonics. Default is 1.
        order_filter (int): Order of the filter spherical harmonics. Default is 1.
        stride (int): Stride of the convolution. Default is 1.
        padding (int): Padding size. Default is 0.
        dilation_rate (int): Dilation rate of the convolution. Default is 1.
        bias (bool): Whether to use bias. Default is True.
        device (torch.device): Device to use for computation. Default is 'cuda' if available, else 'cpu'.

    """
    def __init__(self, num_inputs, num_outputs, kernel_size, order_in=1, order_out=1, order_filter=1,
                 stride=1, padding=0, dilation_rate=1, bias=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(EquiLocalPatOrientConvolution, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.stride = stride
        self.padding = padding
        self.device = device
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "Kernel size should be an odd number."

        self.order_in = order_in
        self.order_out = order_out
        self.order_filter = order_filter
        self.dilation_rate = dilation_rate
        self._initialize_parameters(bias)

    def _initialize_parameters(self, bias):
        """Initialize the parameters of the module."""
        spherical_coords, self.masks = self._compute_spherical_coords_and_masks()
        self.P = self.masks.shape[0]
        self.weight = []
        for l1 in range(self.order_in):
            for l2 in range(self.order_filter):
                self.weight.append(nn.Parameter(torch.randn(self.num_outputs, self.num_inputs,
                                                            2*l1+1, 2*l1+1, 2*l2+1, self.P)))
        self.weight = nn.ParameterList(self.weight)
        self.sph_harm = spherical_harmonics(self.order_filter, spherical_coords[..., 2],
                                            spherical_coords[..., 1], device=self.device)
        if self.order_filter > 1:
            center = (self.kernel_size - 1) // 2
            for l in range(1, self.order_filter):
                self.sph_harm[l][:, center, center, center] = 0.0

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_outputs))
        else:
            self.register_parameter('bias', None)

        self.sh_cg = self._compute_sh_cg()

    def _compute_spherical_coords_and_masks(self):
        """Compute spherical coordinates and masks."""
        kernel_center = (self.kernel_size - 1) // 2
        spherical_coords = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size, 3), dtype=np.float32)
        rad_list = []
        masks = []

        t = (self.kernel_size - 1) // 2 + 1
        k_max = ((t - 1) ** 2) * 3

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    i1, j1, k1 = i - kernel_center, j - kernel_center, k - kernel_center
                    if i1 == 0 and j1 == 0 and k1 == 0:
                        spherical_coords[i, j, k, 1] = 0.0
                        spherical_coords[i, j, k, 2] = 0.0
                        if (self.kernel_size - 1) == 0:
                            spherical_coords[i, j, k, 0] = 0.0
                        else:
                            spherical_coords[i, j, k, 0] = -1.0
                    else:
                        spherical_coords[i, j, k, 0] = i1 ** 2 + j1 ** 2 + k1 ** 2
                        spherical_coords[i, j, k, 0] = (2 * spherical_coords[i, j, k, 0] / k_max - 1.0)
                        spherical_coords[i, j, k, 1] = np.arctan2(np.sqrt(i1 ** 2 + j1 ** 2), k1)
                        spherical_coords[i, j, k, 2] = np.arctan2(j1, i1)

                    if spherical_coords[i, j, k, 0] not in rad_list:
                        rad_list.append(spherical_coords[i, j, k, 0])
                        m = len(rad_list) - 1
                        masks.append(np.zeros((self.kernel_size, self.kernel_size, self.kernel_size), dtype=np.float32))
                    else:
                        m = rad_list.index(spherical_coords[i, j, k, 0])
                    masks[m][i, j, k] = 1.0

        return (torch.tensor(spherical_coords, dtype=torch.float32, device=self.device),
                torch.tensor(np.stack(masks[::-1], axis=0), dtype=torch.float32, device=self.device))

    def _compute_sh_cg(self):
        """Compute the spherical harmonic coefficients."""
        return [[[torch.einsum('mijk, lmn->lnijk', self.sph_harm[l2], clebsch_gordan[l1][l2][l3 - abs(l1 - l2)])
                  for l3 in range(abs(l1 - l2), l1 + l2 + 1)]
                 for l2 in range(self.order_filter)]
                for l1 in range(self.order_in)]

    def _compute_filters(self):
        """Compute the convolution filters."""
        args = (*self.weight, self.sh_cg, self.masks, self.order_in, self.order_out, self.order_filter,
                self.num_outputs, self.num_inputs, self.kernel_size)
        filters = ComputeFiltersFunction.apply(*args)
        return filters

    def _se3_convolution(self, input, filters):
        """Perform SE(3) convolution."""
        input_reshaped = torch.reshape(input, [input.size(0), -1, input.size(3), input.size(4), input.size(5)])
        filter_one_tensor = torch.reshape(
            torch.concat([torch.concat([torch.reshape(filters[l1 * self.order_out + l3],
                                                      [self.num_outputs, (2 * l3 + 1) ** 2, self.num_inputs, (2*l1+1)**2, self.kernel_size, self.kernel_size, self.kernel_size]) for l3 in range(self.order_out)], dim = 1) for l1 in range(self.order_in)], dim = 3), [int(self.num_outputs*self.order_out*(4*self.order_out**2-1)//3), int(self.num_inputs*self.order_in*(4*self.order_in**2-1)//3), self.kernel_size, self.kernel_size, self.kernel_size])
        
        conv = F.conv3d(input_reshaped, filter_one_tensor, stride=self.stride, padding=self.padding, dilation=self.dilation_rate)
        return torch.reshape(conv, [conv.size(0),self.num_outputs,-1,conv.size(2),conv.size(3),conv.size(4)])
        

    
    def forward(self, input):
        """
        Forward pass of the module.

        Parameters:
        input (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the convolution and bias.
        """
        filters = self._compute_filters()
        output = self._se3_convolution(input, filters)
        if self.bias is not None:
            output += torch.cat([self.bias[None, :, None, None, None, None, None], torch.zeros([1, self.num_outputs, self.order_out*(4*self.order_out**2-1)//3-1, 1,1,1])], dim = 2)
        
        return output