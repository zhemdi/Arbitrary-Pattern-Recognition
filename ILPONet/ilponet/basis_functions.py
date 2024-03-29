import torch
import numpy as np

# Import necessary functions from wigner_functions.py
from .wigner_functions import wigner_d_matrix_real0, wigner_d_matrix_real



# get_so3basisgrid0, get_so3basisgrid, associated_legendre_polynomials, and spherical_harmonics


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