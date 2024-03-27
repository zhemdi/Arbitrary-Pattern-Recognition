import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import necessary functions and classes from other files
from wigner_functions import wigner_d_matrix_real
from basis_functions import get_so3basisgrid, get_so3basisgrid0, spherical_harmonics
from wigmat_reconstruction import WigmatReconstruction


EPS = 1e-16

class InvLocalPatOrientConvolution(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size, order=1, so3_size=1, stride=1, padding=0,
                 dilation_rate=1, bias=True, pooling_type='softmax',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), beta_division='regular'):
        super(InvLocalPatOrientConvolution, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.stride = stride
        self.padding = padding
        self.device = device

        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "Kernel size should be an odd number."

        self.order = order
        self.dilation_rate = dilation_rate
        self.pooling_type = pooling_type if pooling_type != 'max' else 'hardmax'
        assert self.pooling_type in ['hardmax', 'softmax'], "Invalid pooling type"

        # Initialize weights, bias and other parameters
        self._initialize_parameters(so3_size, beta_division, bias)
        
    def _initialize_parameters(self, so3_size, beta_division, bias):
        """Helper method to initialize weights and other parameters."""
        t = (self.kernel_size - 1) // 2 + 1
        k_max = ((t - 1) ** 2) * 3

        so3basisgrid_decom_tensors = get_so3basisgrid(self.order, K=so3_size, beta_division=beta_division)
        self.wigner_indices = self._compute_wigner_indices()
        self.spherical_coords, self.masks = self._compute_spherical_coords_and_masks()
        self.sph_harm = spherical_harmonics(self.order, self.spherical_coords[..., 2], self.spherical_coords[..., 1])

        

        # Corrections for order > 1
        if self.order > 1:
            center = (self.kernel_size - 1) // 2
            self.sph_harm[1:, center, center, center] = 0.0

        self.sph_harm_gath = self.sph_harm[self.wigner_indices[:, 2]]
        self.basis_functions = torch.einsum('rijk, lijk->lrijk', self.masks,
                                            self.sph_harm).to(self.device)
        self.d1, self.d2, self.cossin_alpha1, self.cossin_gamma1, self.cossin_alpha2, self.cossin_gamma2, self.w_i = \
            [torch.tensor(t, dtype=torch.float32).to(self.device) for t in so3basisgrid_decom_tensors]
        so3basisgrid, w_i = get_so3basisgrid0(self.order, K = so3_size, beta_division=beta_division)
        self.so3basisgrid = torch.tensor(so3basisgrid, dtype=torch.float32).to(self.device)
        self.w_i = torch.tensor(w_i, dtype=torch.float32).to(self.device)

        # Weight initialization
        self.weight = nn.Parameter(torch.randn(self.order ** 2, len(self.masks) - 1, self.num_inputs, self.num_outputs))
        self.zeroweight = nn.Parameter(torch.randn(self.num_inputs, self.num_outputs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_outputs))
        else:
            self.register_parameter('bias', None)

        self.wigmat_reconstruction = WigmatReconstruction.apply

    def _compute_wigner_indices(self):
        """Helper method to compute Wigner indices."""
        wigner_indices = []
        for l in range(self.order):
            wigner_indices.extend(
                [[l, l**2 + i // (2 * l + 1), l**2 + i % (2 * l + 1), (self.order - 1 - l + i // (2 * l + 1)) *
                  (2 * self.order - 1) + (self.order - 1 - l + i % (2 * l + 1))] for i in range((2 * l + 1) ** 2)])
        return np.array(wigner_indices)

    def _compute_spherical_coords_and_masks(self):
        """Helper method to compute spherical coordinates and masks."""
        kernel_center = (self.kernel_size - 1) // 2
        spherical_coords = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size, 3), dtype=np.float32)
        rad_list = []
        masks = []

        t = (self.kernel_size-1)//2 + 1
        k_max = ((t-1)**2)*3


        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    i1, j1, k1 = i - kernel_center, j - kernel_center, k - kernel_center
                    if i1 == 0 and j1 == 0 and k1 == 0:
                        spherical_coords[i,j,k,1] = 0.0
                        spherical_coords[i,j,k,2] = 0.0
                        if (self.kernel_size-1) == 0:
                            spherical_coords[i,j,k,0] = 0.0
                        else:
                            spherical_coords[i,j,k,0] = -1.0
                    else:
                        spherical_coords[i,j,k,0] = i1**2+j1**2+k1**2
                        spherical_coords[i,j,k,0] = (2*spherical_coords[i,j,k,0]/k_max - 1.0)
                        spherical_coords[i,j,k,1] = np.arctan2(np.sqrt(i1**2+j1**2), k1)
                        spherical_coords[i,j,k,2] = np.arctan2(j1, i1)
                    

                    if spherical_coords[i,j,k,0] not in rad_list:
                        rad_list.append(spherical_coords[i,j,k,0])
                        m = len(rad_list)-1
                        masks.append(np.zeros((self.kernel_size, self.kernel_size, self.kernel_size), dtype = np.float32))
                    else:
                        m = rad_list.index(spherical_coords[i,j,k,0])
                    masks[m][i,j,k] = 1.0

        return torch.tensor(spherical_coords, dtype = torch.float32, device = self.device), torch.tensor(np.stack(masks[::-1], axis = 0), dtype = torch.float32, device = self.device)

    def forward(self, input):
        """Forward pass for the convolution."""
        self.wm_ind = torch.tensor(self.wigner_indices[:, 3], device=self.weight.device)
        zeroweight_ext = torch.cat([self.zeroweight[None, None], torch.zeros(self.order**2 - 1, 1, self.num_inputs,
                                                                        self.num_outputs, device=self.weight.device)],
                                dim=0)
        weight = torch.cat([zeroweight_ext, self.weight], dim=1)
        kernel_in_3d = torch.einsum('lred,lrijk->ledijk', weight[self.wigner_indices[:, 1]], self.basis_functions[self.wigner_indices[:, 2]])
        kernel_in_3d = kernel_in_3d.reshape(-1, self.num_inputs, self.kernel_size, self.kernel_size, self.kernel_size)
        conv = F.conv3d(input, kernel_in_3d, stride=self.stride, padding=self.padding)
        conv_reshaped = conv.reshape(-1, len(self.wigner_indices), self.num_outputs, *conv.shape[2:])
        # so3_distr = WigmatReconstruction.apply([conv_reshaped, self.d1, self.d2, self.cossin_alpha1, self.cossin_gamma1,
        #                                         self.cossin_alpha2, self.cossin_gamma2, self.wm_ind, self.order])
        so3_distr = torch.einsum('bsdxyz, mlns->mlnbdxyz',conv_reshaped, self.so3basisgrid)

        if self.pooling_type == 'softmax':
            so3_distr = F.relu(so3_distr)
            w = so3_distr/((so3_distr*self.w_i[None,:,None,None,None,None,None,None]).sum(0, keepdims = True).sum(1, keepdims = True).sum(2, keepdims = True) + EPS)
            output = (so3_distr*w*self.w_i[None,:,None,None,None,None,None,None]).sum(0).sum(0).sum(0)
        else:
            output = torch.max(so3_distr.reshape(*([-1]+list(so3_distr.shape[3:]))), 0)[0]
        if self.bias is not None:
            output += self.bias.view(1,-1,1,1,1)
        return output