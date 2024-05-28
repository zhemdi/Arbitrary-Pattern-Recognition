import torch
import numpy as np

def associated_legendre_polynomials(L, x):
    """
    Compute the associated Legendre polynomials.

    Parameters:
    L (int): The maximum degree of the polynomials.
    x (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: A tensor containing the associated Legendre polynomials.
    """
    P = [torch.ones_like(x) for _ in range((L+1)*L//2)]
    
    # Compute the polynomials for l in range(1, L)
    for l in range(1, L):
        P[(l+3)*l//2] = - np.sqrt((2*l-1)/(2*l)) * torch.sqrt(1-x**2) * P[(l+2)*(l-1)//2]
    
    # Compute the polynomials for m in range(L-1)
    for m in range(L-1):
        P[(m+2)*(m+1)//2+m] = x * np.sqrt(2*m+1) * P[(m+1)*m//2+m]
        for l in range(m+2, L):
            P[(l+1)*l//2+m] = ((2*l-1)*x*P[l*(l-1)//2 + m]/np.sqrt((l**2-m**2)) - P[(l-1)*(l-2)//2+m]*np.sqrt(((l-1)**2-m**2)/(l**2-m**2)))
    return torch.stack(P, dim=0)

def spherical_harmonics(L, THETA, PHI, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Compute the spherical harmonics.

    Parameters:
    L (int): The maximum degree of the harmonics.
    THETA (torch.Tensor): The theta angles.
    PHI (torch.Tensor): The phi angles.
    device (torch.device): The device to use for computations (default is CUDA if available).

    Returns:
    list: A list of tensors containing the spherical harmonics for each degree l.
    """
    P = associated_legendre_polynomials(L, torch.cos(PHI))
    M2 =  [torch.zeros_like(THETA) for _ in range(2*(L-1)+1)]
    output =  [[torch.zeros_like(THETA, device = device) for _ in range(2*l+1)] for l in range(L)]
    
    # Compute cosine and sine components for each m
    for m in range(L):
        if m > 0:
            M2[L-1+m] = torch.cos(m*THETA)
            M2[L-1-m] = torch.sin(m*THETA)
        else:
            M2[L-1]  = torch.ones_like(THETA)
    
    # Compute the spherical harmonics for each l and m
    for l in range(L):
        for m in range(l+1):
            if m > 0:
                output[l][l+m] = np.sqrt(2)*P[(l+1)*l//2+m]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1+m]
                output[l][l-m] = np.sqrt(2)*P[(l+1)*l//2+m]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1-m]
            else:
                output[l][l  ] = P[(l+1)*l//2]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1]
    
    return [torch.stack(output_i, dim = 0).to(device) for output_i in output]
