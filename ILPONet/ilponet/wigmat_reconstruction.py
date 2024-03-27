import torch


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

