import torch
from .cg_coefficients import CG_tensor


def tensor_product(*args):
    """
    Computes the tensor product of the input tensors.

    Args:
        *args: Input tensors and dimensions (Lx, Ly, Lz).
            - The last three arguments should be Lx, Ly, and Lz.
            - The remaining arguments are the input tensors.

    Returns:
        torch.Tensor: The result of the tensor product.
    """
    Lx, Ly, Lz = args[-3], args[-2], args[-1]

    Lx3 = Lx * (4 * Lx ** 2 - 1) // 3
    Ly3 = Ly * (4 * Ly ** 2 - 1) // 3
    Lz3 = Lz * (4 * Lz ** 2 - 1) // 3
    L3 = min([Lx, Ly, Lz])

    B, N, _, H, W, D = args[0].shape
    CG_tensor_order = CG_tensor[:Lx3, :Ly3, :Lz3]
    CG_tensor_indices = torch.nonzero(CG_tensor_order, as_tuple=False)

    CG_tensor_values = CG_tensor_order[
        CG_tensor_indices[:, 0], CG_tensor_indices[:, 1], CG_tensor_indices[:, 2]
    ]

    output = torch.zeros(
        args[0].size(0),
        args[0].size(1),
        Lz3,
        args[0].size(3),
        args[0].size(4),
        args[0].size(5),
        device=args[0].device,
    )

    for i in range(CG_tensor_indices.size(0) // L3 + 1):
        output_values = (
            CG_tensor_values[None, None, i * L3 : (i + 1) * L3, None, None, None]
            * args[0][:, :, CG_tensor_indices[i * L3 : (i + 1) * L3, 0]]
            * args[1][:, :, CG_tensor_indices[i * L3 : (i + 1) * L3, 1]]
        )

        output.index_add_(2, CG_tensor_indices[i * L3 : (i + 1) * L3, 2], output_values)

    return output