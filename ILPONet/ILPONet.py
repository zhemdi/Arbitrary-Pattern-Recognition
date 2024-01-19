import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import sph_harm

EPS = 1e-16



def get_small_d_wigner_for_real(beta, L_max):
    """
    Generate small Wigner D matrices.
    """
    def index_offset(j):
        return j*(4*j**2-1)//3
    def index_offset2(j, m, mp):
        return index_offset(j) + (2*j+1)*(j+m) + j + mp
    def make_wigner_theta(beta, WignerArray, order):
        beta = np.asarray(beta)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        cos_beta2 = np.cos(beta * 0.5)
        sin_beta2 = np.sin(beta * 0.5)
        tan_beta2 = sin_beta2 / cos_beta2

        if order > 0:
            WignerArray[..., index_offset2(0,0,0)] = 1.0

        if order > 1:
            WignerArray[..., index_offset2(1,0,0)] = cos_beta
            WignerArray[..., index_offset2(1,1,-1)] = sin_beta2 * sin_beta2
            WignerArray[..., index_offset2(1,1,0)] = -sin_beta / np.sqrt(2.0)
            WignerArray[..., index_offset2(1,1,1)] = cos_beta2 * cos_beta2
            WignerArray[..., index_offset2(1,0,1)] = -WignerArray[..., index_offset2(1,1,0)]
            WignerArray[..., index_offset2(1,0,-1)] = WignerArray[..., index_offset2(1,1,0)]
            WignerArray[..., index_offset2(1,-1,-1)] = WignerArray[..., index_offset2(1,1,1)]
            WignerArray[..., index_offset2(1,-1, 1)] = WignerArray[..., index_offset2(1,1,-1)]
            WignerArray[..., index_offset2(1,-1, 0)] = -WignerArray[..., index_offset2(1,1,0)] 

            d1_0_0 = WignerArray[..., index_offset2(1,0,0)]
            d1_1_1 = WignerArray[..., index_offset2(1,1,1)]
        for i in range(2, order):
            two_i_m_1 = i + i - 1
            sq_i = i * i
            sq_i_m_1 = (i - 1) * (i - 1)
            for j in range(i-1):
                sq_j = j * j
                for k in range(-j,j+1):
                    sq_k = k * k
                    a = (i * two_i_m_1)/np.sqrt((sq_i - sq_j)*(sq_i - sq_k))
                    b = (d1_0_0 - ((j*k)/(i*(i-1))))
                    c = np.sqrt((sq_i_m_1 - sq_j)*(sq_i_m_1 - sq_k))/((i-1)*two_i_m_1)
                    WignerArray[..., index_offset2(i, j, k)] = a * (b * WignerArray[..., index_offset2(i - 1, j, k)] - c * WignerArray[..., index_offset2(i - 2, j, k)])
                    WignerArray[..., index_offset2(i, k, j)] = (-1) ** (j - k) * WignerArray[..., index_offset2(i, j, k)]
                    WignerArray[..., index_offset2(i, -j, -k)] = (-1) ** (j - k) * WignerArray[..., index_offset2(i, j, k)]
                    WignerArray[..., index_offset2(i, -k, -j)] = WignerArray[..., index_offset2(i, j, k)]

            WignerArray[..., index_offset2(i,i,i)] = d1_1_1 * WignerArray[..., index_offset2(i-1,i-1,i-1)]
            WignerArray[..., index_offset2(i,-i,-i)] = WignerArray[..., index_offset2(i,i,i)]
            WignerArray[..., index_offset2(i,i-1,i-1)] = (i * d1_0_0 - i + 1) * WignerArray[..., index_offset2(i-1,i-1,i-1)]
            WignerArray[..., index_offset2(i,-i+1,-i+1)] = WignerArray[..., index_offset2(i,i-1,i-1)]
            for minus_k in range(-i,i):
                k = -minus_k
                WignerArray[..., index_offset2(i,i,k-1)] = - np.sqrt((i+k)/(i-k+1)) * tan_beta2 * WignerArray[..., index_offset2(i,i,k)]
                WignerArray[..., index_offset2(i,k-1,i)] = (-1)**(i-k+1)*WignerArray[..., index_offset2(i,i,k-1)]
                WignerArray[..., index_offset2(i,-i,-k+1)] = (-1)**(i-k+1)*WignerArray[..., index_offset2(i,i,k-1)]
                WignerArray[..., index_offset2(i,-k+1,-i)] = WignerArray[..., index_offset2(i,i,k-1)]
            for minus_k in range(1-i,i-1):
                k = -minus_k
                a = np.sqrt((i+k)/((i+i)*(i-k+1)))
                WignerArray[..., index_offset2(i,i-1,k-1)] = (i*cos_beta-k+1) * a * WignerArray[..., index_offset2(i,i,k)] / d1_1_1
                WignerArray[..., index_offset2(i,k-1,i-1)] = (-1)**(i-k-2)*WignerArray[..., index_offset2(i,i-1,k-1)]
                WignerArray[..., index_offset2(i,-i+1,-k+1)] = (-1)**(i-k-2)*WignerArray[..., index_offset2(i,i-1,k-1)]
                WignerArray[..., index_offset2(i,-k+1,-i+1)] = WignerArray[..., index_offset2(i,i-1,k-1)]
            for k in range(1, i+1):
                for j in range(k):
                    phase = (-1)**(j+k)
                    WignerArray[..., index_offset2(i,j,k)] = phase * WignerArray[..., index_offset2(i,k,j)]
                    WignerArray[..., index_offset2(i,j,-k)] =  WignerArray[..., index_offset2(i,k,-j)]
                    WignerArray[..., index_offset2(i,-j,k)] =  phase * WignerArray[..., index_offset2(i,j,-k)]
                    WignerArray[..., index_offset2(i,-j,-k)] =  phase * WignerArray[..., index_offset2(i,j,k)]

    total_size = sum([(2 * j + 1) ** 2 for j in range(L_max)])
    WignerArray = np.zeros(beta.shape + (total_size,) )
    make_wigner_theta(beta, WignerArray, L_max)
    return WignerArray


def wigner_d_matrix_real0(L, alpha, beta, gamma):
    
    d_matrices = get_small_d_wigner_for_real(beta, L)
    d_real_matrices = np.zeros((len(alpha), len(beta), len(gamma), d_matrices.shape[1]))#, dtype = np.float32)
    alpha = np.reshape(alpha, [-1,1,1,1])
    d_matrices = d_matrices[None,:, None, :]
    gamma = np.reshape(gamma, [1,1,-1,1])
    def idx(j, m, n):
        return j*(-1 + 4*j**2)//3 + (2 * j + 1) * m + n

    def get_d(l, m, n):
        return d_matrices[...,idx(l, m + l, n + l)]
        # if m <= n:
        #     return d_matrices[...,idx(l, m + l, n + l)]
        # else:
        #     return (-1)**(m - n) * d_matrices[...,idx(l, n + l, m + l)]
    def cossin(arg, t):
        return np.cos(arg)*(1-t)+np.sin(arg)*t
    
    indices = []
    factors = []
    for l in range(L):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                
                if m == 0 and n == 0:
                    factors.append([1,0])
                    t = 0
                elif m == 0 and n > 0:
                    factors.append([np.sqrt(2),0])
                    t = 0
                elif m == 0 and n < 0:
                    factors.append([(-1)**n*np.sqrt(2),0])
                    t = 1
                elif m > 0 and n == 0:
                    factors.append([np.sqrt(2),0])
                    t = 0
                elif m < 0 and n == 0:
                    factors.append([-(-1)**m*np.sqrt(2),0])
                    t = 1
                elif m > 0 and n > 0:
                    factors.append([1, (-1)**n])
                    t = 0
                elif m > 0 and n < 0:
                    factors.append([(-1)**n, -1])
                    t = 1
                elif m < 0 and n > 0:
                    factors.append([-(-1)**m, - (-1)**(m+n)])
                    t = 1
                elif m < 0 and n < 0:
                    factors.append([(-1)**(m+n), - (-1)**(m)])
                    t = 0
                indices.append([l, m, n, l*(4*l**2-1)//3+(2*l+1)*(l+m)+l-n,t])
                    
           
    indices = np.array(indices, dtype = np.int32)
    factors = np.array(factors)
    d_real_matrices = factors[:,0]*cossin(indices[:,1]*alpha + indices[:,2]*gamma, indices[:,-1])*d_matrices + factors[:,1]*cossin(indices[:,1]*alpha - indices[:,2]*gamma, indices[:,-1])*d_matrices[...,indices[:,3]]

    
    
               

    return d_real_matrices


def wigner_d_matrix_real(L, alpha, beta, gamma):
    """
    Generate real Wigner D matrices.
    """
    d_matrices = get_small_d_wigner_for_real(beta, L)
    def cossin(arg, t):
        return np.cos(arg)*(1-t)+np.sin(arg)*t
    
    cossin_alpha1 = []
    cossin_gamma1 = []
    cossin_alpha2 = []
    cossin_gamma2 = []
    for m in range(-L+1, L):
        cossin_alpha1.append(cossin(m*alpha, 1.0*(m < 0)))
        cossin_gamma1.append(cossin(m*gamma, 1.0*(m < 0)))
        cossin_alpha2.append(cossin(m*alpha, 1.0*(m > 0)))
        cossin_gamma2.append(cossin(m*gamma, 1.0*(m > 0)))

    cossin_alpha1 = np.array(cossin_alpha1)
    cossin_gamma1 = np.array(cossin_gamma1)
    cossin_alpha2 = np.array(cossin_alpha2)
    cossin_gamma2 = np.array(cossin_gamma2)

    
    indices = []
    factors = []

    for l in range(L):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                
                if m == 0 and n == 0:
                    factors.append([1,0,0,0])
                    
                elif m == 0 and n > 0:
                    factors.append([np.sqrt(2),0,0,0])
                    
                elif m == 0 and n < 0:
                    factors.append([(-1)**n*np.sqrt(2),0,0,0])
                    
                elif m > 0 and n == 0:
                    factors.append([np.sqrt(2),0,0,0])
                    
                elif m < 0 and n == 0:
                    factors.append([-(-1)**m*np.sqrt(2),0,0,0])
                    
                elif m > 0 and n > 0:
                    factors.append([1, (-1)**n,-1, (-1)**n])
                    
                elif m > 0 and n < 0:
                    factors.append([(-1)**n, 1, (-1)**n, -1])
                    
                elif m < 0 and n > 0:
                    factors.append([-(-1)**m, - (-1)**(m+n), -(-1)**m, (-1)**(m+n)])
                    
                elif m < 0 and n < 0:
                    factors.append([-(-1)**(m+n), - (-1)**(m), (-1)**(m+n), - (-1)**(m)])
                    
                indices.append(l*(4*l**2-1)//3+(2*l+1)*(l+m)+l-n)
                    
           
    indices = np.array(indices, dtype = np.int32)
    factors = np.array(factors)
    d1 = factors[:,0]*d_matrices + factors[:,1]*d_matrices[...,indices] 
    d2 = factors[:,2]*d_matrices + factors[:,3]*d_matrices[...,indices]
    return d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2



def get_so3basisgrid0(L, K = 100, beta_division = 'regular'):
    assert beta_division in ['regular', 'gauss']
    alpha = np.linspace(0, 2 * np.pi, K, endpoint= False)
    if beta_division == 'regular':
        x_i = -1 + (np.arange(K)+0.5)*2/K  
        w_i = np.ones(K)
    else:
        x_i, w_i = np.polynomial.legendre.leggauss(K)
    beta = np.arccos(x_i)
    gamma = np.linspace(0, 2 * np.pi, K, endpoint= False)
    d_matrix = wigner_d_matrix_real0(L, alpha, beta, gamma)
    return d_matrix, w_i

def get_so3basisgrid(L, K = 100, beta_division = 'regular'):
    """
    Get SO3 basis grid.
    """
    assert beta_division in ['regular', 'gauss']
    alpha = np.linspace(0, 2 * np.pi, K, endpoint= False)
    if beta_division == 'regular':
        x_i = -1 + (np.arange(K)+0.5)*2/K  
        w_i = np.ones(K)
    else:
        x_i, w_i = np.polynomial.legendre.leggauss(K)
    beta = np.arccos(x_i)
    
    gamma = np.linspace(0, 2 * np.pi, K, endpoint= False)
    d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2 = wigner_d_matrix_real(L, alpha, beta, gamma)
    return [d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, w_i]


def associated_legendre_polynomials(L, x):
    """
    Associated Legendre polynomials.
    """
    P = [torch.ones_like(x) for _ in range((L+1)*L//2)]
    for l in range(1, L):
        P[(l+3)*l//2] = - np.sqrt((2*l-1)/(2*l)) * torch.sqrt(1-x**2) * P[(l+2)*(l-1)//2]
    for m in range(L-1):
        P[(m+2)*(m+1)//2+m] = x * np.sqrt(2*m+1) * P[(m+1)*m//2+m]
        for l in range(m+2, L):
            P[(l+1)*l//2+m] = ((2*l-1)*x*P[l*(l-1)//2 + m]/np.sqrt((l**2-m**2)) - P[(l-1)*(l-2)//2+m]*np.sqrt(((l-1)**2-m**2)/(l**2-m**2)))
    return torch.stack(P, dim=0)

def spherical_harmonics(L, THETA, PHI):
    """
    Spherical harmonics.
    """
    P = associated_legendre_polynomials(L, torch.cos(PHI))
    output =  [torch.zeros_like(THETA) for _ in range(L**2)]
    M2 =  [torch.zeros_like(THETA) for _ in range(2*(L-1)+1)]
    for m in range(L):
        if m > 0:
            M2[L-1+m] = torch.cos(m*THETA)
            M2[L-1-m] = torch.sin(m*THETA)
        else:
            M2[L-1]  = torch.ones_like(THETA)
    for l in range(L):
        for m in range(l+1):
            if m > 0:
                output[l**2 +l+m] = np.sqrt(2)*P[(l+1)*l//2+m]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1+m]
                output[l**2+ l-m] = np.sqrt(2)*P[(l+1)*l//2+m]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1-m]
            else:
                output[l**2 +l  ] = P[(l+1)*l//2]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1]
    return torch.stack(output, dim = 0)




class WigmatReconstruction(torch.autograd.Function):
    """
    Reconstruct Wigner matrix.
    """

    @staticmethod
    def forward(ctx, input ):
        
        wm_coefs, d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, wm_ind, order = input
        ctx.save_for_backward(d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, wm_ind, order)
        
        w1 = torch.einsum('ls,bsdxyz->slbdxyz', d1, wm_coefs)
        w2 = torch.einsum('ls,bsdxyz->slbdxyz', d2, wm_coefs)
        w1_reshaped = torch.zeros_like(w1[:(2*order-1)**2], dtype = torch.float32)
        w2_reshaped = torch.zeros_like(w2[:(2*order-1)**2], dtype = torch.float32)
        w1_reshaped.index_add_(0, wm_ind, w1)
        w2_reshaped.index_add_(0, wm_ind, w2)
        w1_reshaped = torch.reshape(w1_reshaped, [2*order-1, 2*order-1]+list(w1.shape[1:]))
        w2_reshaped = torch.reshape(w2_reshaped, [2*order-1, 2*order-1]+list(w2.shape[1:]))
        w1_einsum = torch.einsum('pqlbdxyz,pm->qmlbdxyz', w1_reshaped, cossin_alpha1)
        w1_einsum = torch.einsum('qmlbdxyz,qn->mlnbdxyz', w1_einsum, cossin_gamma1)
        w2_einsum = torch.einsum('pqlbdxyz,pm->qmlbdxyz', w2_reshaped, cossin_alpha2)
        w2_einsum = torch.einsum('qmlbdxyz,qn->mlnbdxyz', w2_einsum, cossin_gamma2)
        output = w1_einsum + w2_einsum
        return output

    @staticmethod
    def backward(ctx, grad_output):
        d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, wm_ind, order = ctx.saved_tensors
        g1 =  torch.einsum('mlnbdxyz,qn-> qmlbdxyz', grad_output, cossin_gamma1)
        g1 =  torch.einsum('qmlbdxyz,pm->pqlbdxyz',  g1,  cossin_alpha1)
        g2 = torch.einsum('mlnbdxyz,qn-> qmlbdxyz', grad_output, cossin_gamma2)
        g2 =  torch.einsum('qmlbdxyz,pm->pqlbdxyz',  g2,  cossin_alpha2)
        g1_reshaped = torch.reshape(g1, [(2*order-1)**2]+list(g1.shape[2:]))[wm_ind]
        g2_reshaped = torch.reshape(g2, [(2*order-1)**2]+list(g2.shape[2:]))[wm_ind]
        grad_wm_coefs =  torch.einsum('ls,slbdxyz->bsdxyz', d1, g1_reshaped) + torch.einsum('ls,slbdxyz->bsdxyz', d2, g2_reshaped) 
        # The input arguments not participating in the computation
        # don't require gradients. So, set their gradients as None.
        grad_d1 = grad_d2 = grad_cossin_alpha1 = grad_cossin_gamma1 = None
        grad_cossin_alpha2 = grad_cossin_gamma2 = grad_wm_ind = grad_order = None

        
        return (grad_wm_coefs, grad_d1, grad_d2, grad_cossin_alpha1, grad_cossin_gamma1, grad_cossin_alpha2, grad_cossin_gamma2, grad_wm_ind, grad_order)




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