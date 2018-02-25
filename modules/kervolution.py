# -* coding=utf-8 *-
# Kervolution Neural Networks
# This file is part of the project of Kervolutional Neural Networks
# It implements the kervolution for 4D tensors in Pytorch
# Copyright (C) 2018 [Wang, Chen] <wang.chen@zoho.com>
# Nanyang Technological University (NTU), Singapore.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import numpy as np


class Kerv2d(nn.Conv2d):
    '''
    kervolution with following options:
    mapping: [translation, polar, logpolar, random]
    kernel_type: [linear, polynomial, sigmoid, gaussian, cauchy]
    default is convolution:
             mapping     --> translation,
             kernel_type --> linear,
    balance, power, slope, gamma is valid only when the kernel_type is specified
    if learnable_kernel = True,  they just be the initial value of learable parameters
    if learnable_kernel = False, they are the value of kernel_type's parameter
    the parameter [power] cannot be learned due to integer limitation
    '''
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            mapping='translation', kernel_type='linear', learnable_kernel=False,
            balance=2, power=4, slope=1, gamma=5):
        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bias_flag, self.stride, self.padding, self.dilation, self.groups = bias, stride, padding, dilation, groups
        self.mapping, self.kernel_type = mapping, kernel_type
        self.kernel_size, self.learnable_kernel = kernel_size, learnable_kernel
        self.in_channels, self.out_channels = in_channels, out_channels
        self.balance, self.power, self.slope, self.gamma = balance, power, slope, gamma

        # parameter for kernel type
        self.weight_ones = Variable(torch.cuda.FloatTensor(self.weight.size()).fill_(1/(self.kernel_size**2)), requires_grad=False)
        if learnable_kernel == True:
            # self.power   = nn.Parameter(torch.cuda.FloatTensor([power]), requires_grad=True) #accuracy 20 epoch 99.090% with init 3.8
            self.balance = nn.Parameter(torch.cuda.FloatTensor([balance]), requires_grad=True)
            self.gamma   = nn.Parameter(torch.cuda.FloatTensor([gamma]), requires_grad=True)
            self.slope   = nn.Parameter(torch.cuda.FloatTensor([slope]), requires_grad=True)

        # mapping functions
        if mapping == 'translation':
            return
        if mapping == 'polar':
            import cv2
            center = (kernel_size/2, kernel_size/2)
            radius = (kernel_size+1) / 2.0
            index = np.reshape(np.arange(kernel_size**2),(kernel_size,kernel_size))
            maps =  np.reshape(cv2.linearPolar(index, center, radius, cv2.WARP_FILL_OUTLIERS).astype(int), (kernel_size**2))
        elif mapping == 'logpolar':
            import cv2
            center = (kernel_size/2, kernel_size/2)
            radius = (kernel_size+1) / 2.0
            M = kernel_size / np.log(radius)
            index = np.reshape(np.arange(kernel_size**2),(kernel_size,kernel_size))
            maps =  np.reshape(cv2.logPolar(index, center, M, cv2.WARP_FILL_OUTLIERS).astype(int), (kernel_size**2))
        elif mapping == 'random':
            maps = np.random.randint(low=0, high=kernel_size**2, size=kernel_size**2)
        else:
            NotImplementedError()
        index_all = np.reshape(np.arange(self.weight.nelement()), (out_channels*in_channels, kernel_size**2))
        index_all[:,:] = index_all[:,maps]
        self.mapping_index = torch.cuda.LongTensor(index_all).view(-1)

    def forward(self, input):
        if self.mapping == 'translation':
            self.weights = self.weight
        else:
            self.weights = self.weight.view(-1)[self.mapping_index]
            self.weights = self.weights.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        y = conv2d(input, self.weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        if self.kernel_type == 'linear':
            return y
        elif self.kernel_type == 'polynomial':
            return (self.slope*y+self.balance) ** self.power
        elif self.kernel_type == 'sigmoid':
            return (self.slope*y+self.balance).tanh()
        elif self.kernel_type == 'gaussian':
            # ignore the weight_norm, which is constant,
            input_norm = conv2d(input**2, self.weight_ones, None, self.stride, self.padding, self.dilation, self.groups)
            return (-self.gamma*(((input_norm-2*y).abs())**2)).exp()
        elif self.kernel_type == 'cauchy':
            # ignore the weight_norm, which is constant, 
            input_norm = conv2d(input**2, self.weight_ones, None, self.stride, self.padding, self.dilation, self.groups)
            return 1/(1+self.gamma*(((input_norm-2*y).abs())**2))
        else:
            return NotImplementedError()
    
    def print_parameters(self):
        if self.learnable_kernel:
            print('power: %.2f, balance: %.2f, slope %.2f, gamma: %.2f' % (
                self.power, self.balance.data[0], self.slope[0], self.gamma.data[0]))


nn.Kerv2d = Kerv2d

if __name__ == '__main__':
    kerv = nn.Kerv2d(in_channels=3,              # input height
                     out_channels=6,             # n_filters
                     kernel_size=5,              # filter size
                     stride=1,                   # filter movement/step
                     padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                     mapping='polar',            # mapping
                     kernel_type='polynomial',   # kernel type
                     learnable_kernel=True)      # enable learning parameters
