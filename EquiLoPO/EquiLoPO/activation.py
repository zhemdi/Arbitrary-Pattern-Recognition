import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .tensor_product import tensor_product

EPS = 1e-12
NORM_FACTOR = 3.0


class CoefficientsNetwork(nn.Module):
    def __init__(self, order=3, num_features=4, hidden_size=4):
        super(CoefficientsNetwork, self).__init__()
        self.weights1 = nn.Parameter(torch.randn(num_features, order, hidden_size))
        self.bias1 = nn.Parameter(torch.randn(num_features, hidden_size))
        self.weights2 = nn.Parameter(torch.randn(num_features, hidden_size, hidden_size))
        self.bias2 = nn.Parameter(torch.randn(num_features, hidden_size))
        self.weights3 = nn.Parameter(torch.randn(num_features, hidden_size, 3))
        self.bias3 = nn.Parameter(torch.randn(num_features, 3))

    def forward(self, x):
        x = torch.einsum('bdxyzk, dkl -> bdxyzl', x, self.weights1) + self.bias1[None, :, None, None, None]
        x = torch.relu(x)
        x = torch.einsum('bdxyzk, dkl -> bdxyzl', x, self.weights2) + self.bias2[None, :, None, None, None]
        x = torch.relu(x)
        x = torch.einsum('bdxyzk, dkl -> bdxyzl', x, self.weights3) + self.bias3[None, :, None, None, None]
        return x / 10


class TrainableActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        coefficients, order = args[-2], args[-1]
        order2 = 2 * order - 1
        l3 = order * (4 * order ** 2 - 1) // 3
        l23 = order2 * (4 * order2 ** 2 - 1) // 3
        x0 = torch.cat([torch.ones_like(args[0][:, :, :1]), torch.zeros(args[0].size(0), args[0].size(1), l23 - 1, args[0].size(3), args[0].size(4), args[0].size(5), device=args[0].device)], dim=2)
        x1 = torch.cat([args[0], torch.zeros(args[0].size(0), args[0].size(1), l23 - l3, args[0].size(3), args[0].size(4), args[0].size(5), device=args[0].device)], dim=2)
        x2_args = (x1, x1, order, order, order2)
        x2 = tensor_product(*x2_args)
        polynomial = coefficients[:, :, None, :, :, :, 0] * x0 + coefficients[:, :, None, :, :, :, 1] * x1 + coefficients[:, :, None, :, :, :, 2] * x2
        polynomial_der = coefficients[:, :, None, :, :, :, 1] * x0 + 2 * coefficients[:, :, None, :, :, :, 2] * x1

        ctx.order = order
        ctx.save_for_backward(*(polynomial_der, x1, x2))
        return polynomial

    @staticmethod
    def backward(ctx, grad_output):
        saved_tensors = ctx.saved_tensors
        order = ctx.order
        order2 = 2 * order - 1
        mult = torch.cat([torch.ones([(2 * l + 1) ** 2], device=grad_output.device) * (2 * l + 1) for l in range(order)], dim=0)
        mult2 = torch.cat([torch.ones([(2 * l + 1) ** 2], device=grad_output.device) * (2 * l + 1) for l in range(order2)], dim=0)

        go = grad_output * mult2[None, None, :, None, None, None]
        args1 = (go, saved_tensors[0], order2, order, order)
        tensor1 = tensor_product(*args1)
        tp = tensor1 / mult[None, None, :, None, None, None]

        der_coef_0 = grad_output[:, :, 0]
        der_coef_1 = (grad_output * saved_tensors[1]).sum(dim=2)
        der_coef_2 = (grad_output * saved_tensors[2]).sum(dim=2)

        grad_outputs = tuple([tp])
        grad_outputs += tuple([torch.stack([der_coef_0, der_coef_1, der_coef_2], axis=-1), None])
        return grad_outputs


class SO3LocalActivation(nn.Module):
    def __init__(self, order, distr_dependency=False, num_feat=4, coefficients_type = 'trainable'):
        super(SO3LocalActivation, self).__init__()
        self.order = order
        self.distr_dependency = distr_dependency
        self.coefficients_type = coefficients_type
        if self.distr_dependency:
            self.coefficients_network = CoefficientsNetwork(order, num_features=num_feat)
            self.mults = nn.Parameter(torch.ones(num_feat))
        else:
            if self.coefficients_type == 'trainable':
                self.coefficients = nn.Parameter(torch.rand(3))
            else:
                self.coefficients = 0.0
        self.coefficients_bias = torch.tensor([3 / 32, 0.5, 15 / 32])

    def forward(self, input):
        with torch.no_grad():
            degree_mult = torch.cat([torch.ones([(2 * l + 1) ** 2], device=input.device) * (2 * l + 1) for l in range(self.order)], dim=0)
            norm = torch.sum(8 * np.pi ** 2 * input ** 2 / degree_mult[None, None, :, None, None, None], dim=2, keepdim=True)
            norm = torch.sqrt(norm + EPS) / NORM_FACTOR

        if self.distr_dependency:
            x = input / (norm * self.mults[None, :, None, None, None, None])
            input_stat = torch.stack([torch.sum(x[:, :, l * (4 * l ** 2 - 1) // 3:(l + 1) * (4 * (l + 1) ** 2 - 1) // 3] ** 2, dim=[2]) for l in range(self.order)], axis=-1)
            coefficients = self.coefficients_network(input_stat) + self.coefficients_bias.to(x.device)[None, None, None, None, None, :]
        elif self.coefficients_type == 'adaptive':
            with torch.no_grad():
                k = input[:, :, 0, :, :, :]
                var = torch.sqrt((input[:, :, 1:] ** 2 / degree_mult[None, None, 1:, None, None, None]).sum(dim=2))
                n = var * NORM_FACTOR + EPS
                kn = k / n

                c0 = (15 / 32 * kn ** 6 - 27 / 32 * kn ** 4 + 9 / 32 * kn ** 2 + 3 / 32)
                c1 = (-15 / 16 * kn ** 5 + 13 / 8 * kn ** 3 - 3 / 16 * kn + 1 / 2)
                c2 = (15 / 32 * kn ** 4 - 15 / 16 * kn ** 2 + 15 / 32)
                coefficients_ad = torch.stack([c0, c1, c2], dim=-1)
                x_min = k - n
                x_max = k + n

                

                coefficients = torch.zeros_like(coefficients_ad)
                condition_coef_ad = (x_min < 0) & (x_max > 0)
                condition_coef_ad_expanded = condition_coef_ad.unsqueeze(-1).expand_as(coefficients_ad)

                condition_negative = (x_max <= 0)
                condition_negative_expanded = condition_negative.unsqueeze(-1).expand_as(coefficients_ad)

                condition_positive = (x_min >= 0)
                condition_positive_expanded = condition_positive.unsqueeze(-1).expand_as(coefficients_ad)

                coefficients = torch.where(condition_coef_ad_expanded, coefficients_ad, coefficients)
                coefficients = torch.where(condition_negative_expanded, torch.tensor([0, 0.01, 0], device=input.device)[None, None, None, None, None, :].expand_as(coefficients_ad), coefficients)
                coefficients = torch.where(condition_positive_expanded, torch.tensor([0, 1, 0], device=input.device)[None, None, None, None, None, :].expand_as(coefficients_ad), coefficients)

                

            x = input / n.unsqueeze(2)
        else:
            x = input / norm
            coefficients = (self.coefficients + self.coefficients_bias.to(x.device))[None, None, None, None, None, :] * torch.ones(x[:, :, 0, :, :, :, None].shape, device=x.device)

        args = (x, coefficients, self.order)
        output = TrainableActivation.apply(*args)

        

        if self.distr_dependency:
            output = output * (norm * self.mults[None, :, None, None, None, None])
        elif self.coefficients_type == 'adaptive':
            output = output * n.unsqueeze(2)

            
        else:
            output = output * norm
        
            

        return output
    
