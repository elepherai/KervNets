# -* coding=utf-8 *-
# Kernel Convolution Networks
# Stride left

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d


class Kerv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bias_flag, self.stride, self.padding, self.dilation, self.groups = bias, stride, padding, dilation, groups
        self.learnable = True
        if self.learnable == True:
            self.poly_d = nn.Parameter(torch.cuda.FloatTensor([2]), requires_grad=True)
            self.balance = nn.Parameter(torch.cuda.FloatTensor([1]), requires_grad=True)
        else:
            self.poly_d = 2
            self.balance = 1

    def forward(self, input):
        if self.bias_flag == True:
            y = conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            y = conv2d(input, self.weight,      None, self.stride, self.padding, self.dilation, self.groups)
        return (y+self.balance) ** self.poly_d

nn.Kerv2d = Kerv2d

if __name__ == '__main__':
    pass
