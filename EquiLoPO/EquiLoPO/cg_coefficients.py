import torch
import e3nn.o3


def _so3_clebsch_gordan(l1, l2, l3):
    """
    Compute the SO(3) Clebsch-Gordan coefficients.

    Parameters:
    l1 (int): The angular momentum quantum number of the first representation.
    l2 (int): The angular momentum quantum number of the second representation.
    l3 (int): The angular momentum quantum number of the third representation.

    Returns:
    torch.Tensor: The Clebsch-Gordan coefficients.
    """
    # Change basis from real to complex for each angular momentum quantum number
    Q1 = e3nn.o3._wigner.change_basis_real_to_complex(l1, dtype=torch.float64)
    Q2 = e3nn.o3._wigner.change_basis_real_to_complex(l2, dtype=torch.float64)
    Q3 = e3nn.o3._wigner.change_basis_real_to_complex(l3, dtype=torch.float64)
    
    # Compute Clebsch-Gordan coefficients
    C = e3nn.o3._wigner._su2_clebsch_gordan(l1, l2, l3).to(dtype=torch.complex128)
    C = torch.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, torch.conj(Q3.T), C)
    
    # Ensure the result is real within a tolerance
    assert torch.all(torch.abs(torch.imag(C)) < 1e-5), "Imaginary part detected beyond tolerance"
    C = torch.real(C)
    return C


class ClebschGordanSingleton:
    """
    Singleton class to store and provide access to Clebsch-Gordan coefficients.
    """
    _instance = None

    def __new__(cls, order_in, order_filter):
        """
        Create and return a singleton instance of ClebschGordanSingleton.

        Parameters:
        order_in (int): The order of the input representation.
        order_filter (int): The order of the filter representation.

        Returns:
        ClebschGordanSingleton: The singleton instance.
        """
        if cls._instance is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls._instance = super(ClebschGordanSingleton, cls).__new__(cls)
            cls._instance.coefficients = [
                [
                    [_so3_clebsch_gordan(l1, l2, l3).float().to(device)
                     for l3 in range(abs(l1 - l2), l1 + l2 + 1)]
                    for l2 in range(order_filter)
                ]
                for l1 in range(order_in)
            ]
        return cls._instance

# Set global angular momentum limit
L_global = 5

# Initialize singleton for Clebsch-Gordan coefficients
cg_singleton = ClebschGordanSingleton(L_global, L_global)
clebsch_gordan = cg_singleton.coefficients


# Create tensor for Clebsch-Gordan coefficients
CG_tensor = []
for l1 in range(L_global):
    CG_tensor_i = []
    for l2 in range(L_global):
        CG_tensor_ij = []
        for l3 in range(2 * L_global - 1):
            if l3 < abs(l1 - l2) or l3 > l1 + l2:
                CG_tensor_ijk = torch.zeros(((2 * l1 + 1) ** 2, (2 * l2 + 1) ** 2, (2 * l3 + 1) ** 2), device=clebsch_gordan[0][0][0].device)
            else:
                cg2 = torch.einsum('ikh,jlf->ijklhf', clebsch_gordan[l1][l2][l3 - abs(l1 - l2)], clebsch_gordan[l1][l2][l3 - abs(l1 - l2)])
                CG_tensor_ijk = torch.reshape(cg2, ((2 * l1 + 1) ** 2, (2 * l2 + 1) ** 2, (2 * l3 + 1) ** 2))
            CG_tensor_ij.append(CG_tensor_ijk)
        CG_tensor_i.append(torch.cat(CG_tensor_ij, dim=2))
    CG_tensor.append(torch.cat(CG_tensor_i, dim=1))
CG_tensor = torch.cat(CG_tensor, dim=0)
