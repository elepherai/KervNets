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
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            mapping='translation', kernel='linear', learnable = True):
        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bias_flag, self.stride, self.padding, self.dilation, self.groups = bias, stride, padding, dilation, groups
        self.mapping, self.kernel = mapping, kernel
        self.kernel_size, self.learnable = kernel_size, learnable

        self.weight_ones = Variable(torch.cuda.FloatTensor(self.weight.size()).fill_(1/(self.kernel_size**2)), requires_grad=False)
        
        if learnable == True:
            self.power = nn.Parameter(torch.cuda.FloatTensor([3.8]), requires_grad=True) #accuracy 98.740 20 epoch 99.090%
            self.balance = nn.Parameter(torch.cuda.FloatTensor([1.6]), requires_grad=True)
            self.gamma = nn.Parameter(torch.cuda.FloatTensor([5]), requires_grad=True)
        else:
            self.power = 3
            self.balance = 2
            self.gamma = 5

    def forward(self, input):
        y = conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.kernel == 'linear':
            return y
        elif self.kernel == 'polynomial':
            return (y+self.balance) ** self.power
        elif self.kernel == 'gaussian':
            input_norm = conv2d(input**2, self.weight_ones, None, self.stride, self.padding, self.dilation, self.groups)
            return (-self.gamma*((input_norm-2*y)**2)).exp()
        else:
            return NotImplementedError()
    def print_parameters(self):
        if self.learnable:
            print('power: %.2f, balance: %.2f, gamma: %.2f' % (self.power,self.balance.data[0], self.gamma.data[0]))


nn.Kerv2d = Kerv2d

if __name__ == '__main__':
    pass
