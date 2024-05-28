import torch
import numpy as np
from .cg_coefficients import clebsch_gordan

class ComputeFiltersFunction(torch.autograd.Function):
    """
    Custom autograd function to compute filters using Clebsch-Gordan coefficients and spherical harmonics.
    """
    @staticmethod
    def forward(ctx, *args):
        """
        Forward pass to compute the filters.

        Parameters:
        ctx (torch.autograd.Function): Context object to save information for backward pass.
        args: A list of input arguments including weight tensors, spherical harmonics, masks, and other parameters.

        Returns:
        tuple: A tuple containing the computed filters.
        """
        # Save context variables
        ctx.kernel_size = args[-1]
        ctx.num_inputs = args[-2]
        ctx.num_outputs = args[-3]
        ctx.order_filter = args[-4]
        ctx.order_out = args[-5]
        ctx.order_in = args[-6]
        ctx.masks = args[-7]
        ctx.sh_cg = args[-8]
        


        filters = []
        for l1 in range(ctx.order_in):
            for l3 in range(ctx.order_out):
                # Initialize stacked filter with zeros
                stacked_filter = torch.zeros(ctx.num_outputs, (2*l3+1), (2*l3+1), ctx.num_inputs, (2*l1+1), (2*l1+1), ctx.kernel_size, ctx.kernel_size, ctx.kernel_size, device = args[0].device)
                for l2 in range(abs(l1-l3), min(l1+l3+1, ctx.order_filter)):
                    sh = ctx.sh_cg[l1][l2][l3-abs(l1-l2)]
                    cg = clebsch_gordan[l1][l2][l3-abs(l1-l2)]*8*np.pi**2/(2*l1+1)
                    weight_l1_l2 = args[l1*ctx.order_filter + l2]
                    
                    # Compute tensor product of weight and Clebsch-Gordan coefficients
                    tp_cg_l3 = torch.einsum('efslmp, lmd->efsdp', weight_l1_l2, cg)
                    
                    # Apply spherical harmonics and masks
                    d = torch.einsum('efsdp, lnijk, pijk ->ednfslijk', tp_cg_l3, sh, ctx.masks)
                    stacked_filter += d
                
                reshaped_filter = torch.reshape(stacked_filter, [ctx.num_outputs*(2*l3+1)**2, ctx.num_inputs*(2*l1+1)**2, ctx.kernel_size, ctx.kernel_size, ctx.kernel_size])
                filters.append(reshaped_filter)
        return (*filters,)

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Backward pass to compute gradients of the weights.

        Parameters:
        ctx (torch.autograd.Function): Context object with saved information from the forward pass.
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
        tuple: Gradients of the loss with respect to the input arguments.
        """
        order_in = ctx.order_in
        order_out = ctx.order_out
        order_filter = ctx.order_filter
        num_outputs = ctx.num_outputs
        num_inputs = ctx.num_inputs
        kernel_size = ctx.kernel_size
        masks = ctx.masks
        sh_cg = ctx.sh_cg

        E = num_outputs
        F = num_inputs
        P,_,_,_ =  masks.shape

        grad_weight = []
        for l1 in range(order_in):
            L = 2*l1+1
            S = 2*l1+1
            for l2 in range(order_filter):
                M = 2*l2+1
                _, _, I, J, K = grad_output[l1*order_out].shape
                
                grad_weight_l1_l2 =  torch.zeros([E,F,S,L,M,P], device = grad_output[l1*order_out].device)
                for l3 in range(abs(l1-l2), min(l1+l2+1, order_out)):
                    sh = sh_cg[l1][l2][l3-abs(l1-l2)]
                    grad_output_l1_l3 = grad_output[l1*order_out + l3]
                    
                    reshaped_grad_output = torch.reshape(grad_output_l1_l3, [num_outputs, (2*l3+1), (2*l3+1), num_inputs, (2*l1+1), (2*l1+1), kernel_size, kernel_size, kernel_size])
                    grad_stacked_filter = torch.einsum('ednfslijk, lnijk, pijk -> efsdp', reshaped_grad_output, sh, masks)
                    cg = clebsch_gordan[l1][l2][l3-abs(l1-l2)]*8*np.pi**2/(2*l1+1)
                    d = torch.einsum('efsdp, lmd->efslmp', grad_stacked_filter, cg)
                    grad_weight_l1_l2 += d
                grad_weight.append(grad_weight_l1_l2)

            

        grad_outputs = tuple(grad_weight)
        grad_outputs += tuple([None, None, None, None, None, None, None, None])
        return grad_outputs