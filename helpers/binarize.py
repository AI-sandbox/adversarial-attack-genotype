import torch
from torch.autograd import Function
import math

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None
    

def binarize(x):
    x = BinaryQuantize().apply(x, torch.ones(1).to(x.device), torch.ones(1).to(x.device))
    x = (x + 1) / 2
    return x