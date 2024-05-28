import numpy as np

# get_small_d_wigner_for_real, wigner_d_matrix_real0, and wigner_d_matrix_real 


# Function to generate small Wigner D matrices
def get_small_d_wigner_for_real(beta, L_max):
    """
    Generate small Wigner D matrices for given beta and L_max.

    Parameters:
    beta (numpy.ndarray): Array of beta angles.
    L_max (int): Maximum degree of the Wigner D matrices.

    Returns:
    numpy.ndarray: Array of small Wigner D matrices.
    """
    def index_offset(j):
        # Helper function to calculate the index offset for a given j
        return j*(4*j**2-1)//3
    
    def index_offset2(j, m, mp):
        # Helper function to calculate the specific index offset
        return index_offset(j) + (2*j+1)*(j+m) + j + mp
    
    def make_wigner_theta(beta, WignerArray, order):
        """
        Populate WignerArray with values based on beta and the order.

        Parameters:
        beta (numpy.ndarray): Array of beta angles.
        WignerArray (numpy.ndarray): Array to store the Wigner D matrices.
        order (int): Order of the Wigner D matrices.
        """
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

# Function to generate real Wigner D matrices with initial configuration
def wigner_d_matrix_real0(L, alpha, beta, gamma):
    """
    Generate initial real Wigner D matrices.

    Parameters:
    L (int): Maximum degree of the Wigner D matrices.
    alpha (numpy.ndarray): Array of alpha angles.
    beta (numpy.ndarray): Array of beta angles.
    gamma (numpy.ndarray): Array of gamma angles.

    Returns:
    numpy.ndarray: Array of real Wigner D matrices.
    """
    d_matrices = get_small_d_wigner_for_real(beta, L)
    d_real_matrices = np.zeros((len(alpha), len(beta), len(gamma), d_matrices.shape[1]))#, dtype = np.float32)
    
    # Reshape angles for broadcasting
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

    # Calculate indices and factors for Wigner D matrices
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


# Function to generate decomposed real Wigner D matrices
def wigner_d_matrix_real(L, alpha, beta, gamma):
    """
    Generate real Wigner D matrices with detailed components.

    Parameters:
    L (int): Maximum degree of the Wigner D matrices.
    alpha (numpy.ndarray): Array of alpha angles.
    beta (numpy.ndarray): Array of beta angles.
    gamma (numpy.ndarray): Array of gamma angles.

    Returns:
    tuple: Detailed components of the real Wigner D matrices.
    """
    d_matrices = get_small_d_wigner_for_real(beta, L)
    def cossin(arg, t):
        return np.cos(arg)*(1-t)+np.sin(arg)*t
    
    cossin_alpha1 = []
    cossin_gamma1 = []
    cossin_alpha2 = []
    cossin_gamma2 = []


    # Calculate cosine-sine combinations
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

    # Calculate indices and factors for detailed Wigner D matrices
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

    # Compute detailed Wigner D matrices
    d1 = factors[:,0]*d_matrices + factors[:,1]*d_matrices[...,indices] 
    d2 = factors[:,2]*d_matrices + factors[:,3]*d_matrices[...,indices]
    return d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2

