import torch
import torch.nn as nn


class Retyper(nn.Module):
    """
    Retyper module for re-typing input planes.

    Args:
        in_planes (int): Number of input planes.
        out_planes (int): Number of output planes.
        order (int): Range for l in the shape.
    """

    def __init__(self, in_planes, out_planes, order):
        super(Retyper, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.L = order

        self.weight = nn.Parameter(torch.randn(out_planes, in_planes))

    def forward(self, x):
        """
        Forward pass for the Retyper module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, 2l+1, 2l+1, H, W, D).

        Returns:
            torch.Tensor: Output with re-typed planes.
        """
        return torch.einsum('bndijk,pn->bpdijk', x, self.weight)


class SE3BatchNorm(nn.Module):
    """
    SE(3) equivariant batch normalization module.

    Args:
        num_outputs (int): Number of output channels.
        order (int): Range for l in the shape.
        eps (float, optional): Small value for numerical stability. Default: 1e-32.
        momentum (float, optional): Momentum for running statistics. Default: 0.1.
    """

    def __init__(self, num_outputs, order, eps=1e-32, momentum=0.1):
        super(SE3BatchNorm, self).__init__()
        self.num_outputs = num_outputs
        self.L = order
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_outputs).view(1, num_outputs, 1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_outputs).view(1, num_outputs, 1, 1, 1, 1))

        self.register_buffer('running_mean', torch.zeros(num_outputs).view(1, num_outputs, 1, 1, 1, 1))
        self.register_buffer('running_variance', torch.ones(num_outputs).view(1, num_outputs, 1, 1, 1, 1))

    def forward(self, x):
        """
        Forward pass for the SE3BatchNorm module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, 2l+1, 2l+1, H, W, D).

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        use_running_stats = not self.training

        mean = x[:, :, :1].mean(dim=[0, 3, 4, 5], keepdim=True)

        degree_mult = torch.cat([torch.ones([(2*l+1)**2], device=x.device)*(2*l+1) for l in range(self.L)], dim=0)
        mean_sq = torch.sum(x**2 / degree_mult[None, None, :, None, None, None], dim=2, keepdim=True).mean(dim=[0, 3, 4, 5], keepdim=True)

        variance = mean_sq - mean**2

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance

        beta_extended = torch.cat([self.beta, torch.zeros([1, self.num_outputs, self.L*(4*self.L**2-1)//3-1, 1, 1, 1], device=self.beta.device)], dim=2)

        if use_running_stats:
            running_mean_extended = torch.cat([self.running_mean, torch.zeros([1, self.num_outputs, self.L*(4*self.L**2-1)//3-1, 1, 1, 1], device=self.beta.device)], dim=2)
            normalized_output = self.gamma * (x - running_mean_extended) / torch.sqrt(self.running_variance + self.eps) + beta_extended
        else:
            mean_extended = torch.cat([mean, torch.zeros([1, self.num_outputs, self.L*(4*self.L**2-1)//3-1, 1, 1, 1], device=self.beta.device)], dim=2)
            normalized_output = self.gamma * (x - mean_extended) / torch.sqrt(variance + self.eps) + beta_extended

        return normalized_output


class SE3Dropout(nn.Module):
    """
    SE(3) equivariant dropout module.

    Args:
        p (float, optional): Dropout probability. Default: 0.5.
        inplace (bool, optional): If set to True, will do this operation in-place. Default: False.
        num_channels (int, optional): Number of channels. Default: 4.
    """

    def __init__(self, p=0.5, inplace=False, num_channels=4):
        super(SE3Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.num_channels = num_channels

    def forward(self, x):
        """
        Forward pass for the SE3Dropout module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, N, 2l+1, 2l+1, H, W, D).

        Returns:
            torch.Tensor: Output tensor with dropout applied.
        """
        if not self.training or self.p == 0:
            return x

        mask = torch.rand((x.size(0), self.num_channels, 1, x.size(3), x.size(4), x.size(5)), device=x.device) > self.p

        return x * mask

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.p, self.inplace)