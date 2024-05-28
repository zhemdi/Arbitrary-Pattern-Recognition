import torch


class WigmatReconstruction(torch.autograd.Function):
    """
    Reconstruct Wigner matrix using the provided coefficients and basis functions.
    """


    @staticmethod
    def forward(ctx, input ):
        """
        Forward pass for Wigner matrix reconstruction.

        Parameters:
        ctx (torch.autograd.Function): Context object to save information for backward pass.
        input (tuple): Tuple containing the inputs needed for reconstruction:
            wm_coefs (torch.Tensor): Wigner matrix coefficients.
            d1 (torch.Tensor): First set of Wigner D-matrix components.
            d2 (torch.Tensor): Second set of Wigner D-matrix components.
            cossin_alpha1 (torch.Tensor): Cosine-sine combination for alpha1.
            cossin_gamma1 (torch.Tensor): Cosine-sine combination for gamma1.
            cossin_alpha2 (torch.Tensor): Cosine-sine combination for alpha2.
            cossin_gamma2 (torch.Tensor): Cosine-sine combination for gamma2.
            wm_ind (torch.Tensor): Indices for Wigner matrix reconstruction.
            order (int): Order of the Wigner D-matrix.

        Returns:
        torch.Tensor: Reconstructed Wigner matrix.
        """
        wm_coefs, d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, wm_ind, order = input
        ctx.save_for_backward(d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, wm_ind, order)
        
        # Perform tensor contractions to reconstruct the Wigner matrix
        w1 = torch.einsum('ls,bsdxyz->slbdxyz', d1, wm_coefs)
        w2 = torch.einsum('ls,bsdxyz->slbdxyz', d2, wm_coefs)

        # Initialize reshaped tensors with zeros
        w1_reshaped = torch.zeros_like(w1[:(2*order-1)**2], dtype = torch.float32)
        w2_reshaped = torch.zeros_like(w2[:(2*order-1)**2], dtype = torch.float32)

        # Accumulate values at specified indices
        w1_reshaped.index_add_(0, wm_ind, w1)
        w2_reshaped.index_add_(0, wm_ind, w2)

        # Reshape tensors
        w1_reshaped = torch.reshape(w1_reshaped, [2*order-1, 2*order-1]+list(w1.shape[1:]))
        w2_reshaped = torch.reshape(w2_reshaped, [2*order-1, 2*order-1]+list(w2.shape[1:]))

        # Perform additional tensor contractions with cosine-sine combinations
        w1_einsum = torch.einsum('pqlbdxyz,pm->qmlbdxyz', w1_reshaped, cossin_alpha1)
        w1_einsum = torch.einsum('qmlbdxyz,qn->mlnbdxyz', w1_einsum, cossin_gamma1)
        w2_einsum = torch.einsum('pqlbdxyz,pm->qmlbdxyz', w2_reshaped, cossin_alpha2)
        w2_einsum = torch.einsum('qmlbdxyz,qn->mlnbdxyz', w2_einsum, cossin_gamma2)
        
        # Combine the results
        output = w1_einsum + w2_einsum
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Wigner matrix reconstruction.

        Parameters:
        ctx (torch.autograd.Function): Context object with saved information from the forward pass.
        grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
        tuple: Gradients of the loss with respect to the input arguments.
        """
        d1, d2, cossin_alpha1, cossin_gamma1, cossin_alpha2, cossin_gamma2, wm_ind, order = ctx.saved_tensors
        
        # Compute gradients with respect to the inputs
        g1 =  torch.einsum('mlnbdxyz,qn-> qmlbdxyz', grad_output, cossin_gamma1)
        g1 =  torch.einsum('qmlbdxyz,pm->pqlbdxyz',  g1,  cossin_alpha1)
        g2 = torch.einsum('mlnbdxyz,qn-> qmlbdxyz', grad_output, cossin_gamma2)
        g2 =  torch.einsum('qmlbdxyz,pm->pqlbdxyz',  g2,  cossin_alpha2)
        
        # Reshape gradients
        g1_reshaped = torch.reshape(g1, [(2*order-1)**2]+list(g1.shape[2:]))[wm_ind]
        g2_reshaped = torch.reshape(g2, [(2*order-1)**2]+list(g2.shape[2:]))[wm_ind]
        
        # Compute gradient of the Wigner matrix coefficients
        grad_wm_coefs =  torch.einsum('ls,slbdxyz->bsdxyz', d1, g1_reshaped) + torch.einsum('ls,slbdxyz->bsdxyz', d2, g2_reshaped) 
        
        # The input arguments not participating in the computation
        # don't require gradients. So, set their gradients as None.
        grad_d1 = grad_d2 = grad_cossin_alpha1 = grad_cossin_gamma1 = None
        grad_cossin_alpha2 = grad_cossin_gamma2 = grad_wm_ind = grad_order = None

        
        return (grad_wm_coefs, grad_d1, grad_d2, grad_cossin_alpha1, grad_cossin_gamma1, grad_cossin_alpha2, grad_cossin_gamma2, grad_wm_ind, grad_order)

