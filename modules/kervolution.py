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
            mapping='translation', kernel_type='linear', learnable = False):
        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bias_flag, self.stride, self.padding, self.dilation, self.groups = bias, stride, padding, dilation, groups
        self.mapping, self.kernel_type = mapping, kernel_type
        self.kernel_size, self.learnable = kernel_size, learnable
        self.in_channels, self.out_channels = in_channels, out_channels

        self.weight_ones = Variable(torch.cuda.FloatTensor(self.weight.size()).fill_(1/(self.kernel_size**2)), requires_grad=False)
        
        if mapping == 'translation':
            pass
        elif mapping == 'polar':
            import cv2
            import numpy as np
            center = (kernel_size/2, kernel_size/2)
            radius = (kernel_size+1) / 2.0
            index = np.reshape(np.arange(kernel_size*kernel_size),(kernel_size,kernel_size))
            index_all = np.reshape(np.arange(self.weight.nelement()), (out_channels*in_channels, kernel_size*kernel_size))
            maps = torch.from_numpy(cv2.linearPolar(index, center, radius, cv2.WARP_FILL_OUTLIERS).astype(int)).view(kernel_size*kernel_size)
            index_all[:,:] = index_all[:,maps]
            self.mapping_index = torch.cuda.LongTensor(index_all).view(-1)
        elif mapping == 'logpolar':
            import cv2
            import numpy as np
            center = (kernel_size/2, kernel_size/2)
            radius = (kernel_size+1) / 2.0
            M = kernel_size / np.log(radius)
            index = np.reshape(np.arange(kernel_size*kernel_size),(kernel_size,kernel_size))
            index_all = np.reshape(np.arange(self.weight.nelement()), (out_channels*in_channels, kernel_size*kernel_size))
            maps = torch.from_numpy(cv2.logPolar(index, center, M, cv2.WARP_FILL_OUTLIERS).astype(int)).view(kernel_size*kernel_size)
            index_all[:,:] = index_all[:,maps]
            self.mapping_index = torch.cuda.LongTensor(index_all).view(-1)           
        else:
            NotImplementedError()

        self.power = 4

        if learnable == True:
            # self.power  = nn.Parameter(torch.cuda.FloatTensor([3.8]), requires_grad=True) #accuracy 98.740 20 epoch 99.090%
            self.balance = nn.Parameter(torch.cuda.FloatTensor([1.6]), requires_grad=True)
            self.gamma = nn.Parameter(torch.cuda.FloatTensor([5]), requires_grad=True)
        else:
            self.balance = 2
            self.gamma = 5

    def forward(self, input):
        if self.mapping =='translation':
            self.weights = self.weight
        else:
            self.weights = self.weight.view(-1)[self.mapping_index]
            self.weights = self.weights.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        y = conv2d(input, self.weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        if self.kernel_type == 'linear':
            return y
        elif self.kernel_type == 'polynomial':
            return (y+self.balance) ** self.power
        elif self.kernel_type == 'gaussian':
            # ignore the weight_norm, which is constant, 
            input_norm = conv2d(input**2, self.weight_ones, None, self.stride, self.padding, self.dilation, self.groups)
            return (-self.gamma*(((input_norm-2*y).abs())**2)).exp()
        else:
            return NotImplementedError()
    
    def print_parameters(self):
        if self.learnable:
            print('power: %.2f, balance: %.2f, gamma: %.2f' % (self.power, self.balance.data[0], self.gamma.data[0]))


nn.Kerv2d = Kerv2d

if __name__ == '__main__':
    kerv = nn.Kerv2d(in_channels=3,              # input height
                     out_channels=6,             # n_filters
                     kernel_size=5,              # filter size
                     stride=1,                   # filter movement/step
                     padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                     kernel_type='polynomial',
                     mapping='polar',
                     learnable=True)
