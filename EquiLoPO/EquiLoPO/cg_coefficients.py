import torch
import e3nn.o3


def _so3_clebsch_gordan(l1, l2, l3):
    # Change basis from real to complex
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
    _instance = None

    def __new__(cls, order_in, order_filter):
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


L_global = 5
cg_singleton = ClebschGordanSingleton(L_global, L_global)
clebsch_gordan = cg_singleton.coefficients

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
