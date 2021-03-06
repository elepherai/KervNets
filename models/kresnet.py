'''KResNet in PyTorch.

For Pre-activation KResNet, see 'preact_Kresnet.py'.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys
sys.path.append("..")
import kervolution

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.kerv1 = nn.Kerv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.kerv2 = nn.Kerv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Kerv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.kerv1(x)))
        out = self.bn2(self.kerv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.kerv1 = nn.Kerv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.kerv2 = nn.Kerv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.kerv3 = nn.Kerv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Kerv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.kerv1(x)))
        out = F.relu(self.bn2(self.kerv2(out)))
        out = self.bn3(self.kerv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class KResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(KResNet, self).__init__()
        self.in_planes = 64

        # self.kerv1 = nn.Kerv2d(3, 64, 7, 1, 2,
        self.kerv1 = nn.Kerv2d(3, 64, 3, 1,
                        kernel_type='polynomial',
                        learnable_kernel=True,
                        kernel_regularizer=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.kerv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def KResNet18(num_classes=10):
    return KResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def KResNet32(num_classes=10):
    return KResNet(BasicBlock, [3,4,5,3], num_classes=num_classes)

def KResNet34(num_classes=10):
    return KResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def KResNet42(num_classes=10):
    return KResNet(BasicBlock, [4,5,7,4], num_classes=num_classes)

def KResNet52(num_classes=10):
    return KResNet(BasicBlock, [5,7,8,5], num_classes=num_classes)

def KResNet50(num_classes=10):
    return KResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def KResNet101(num_classes=10):
    return KResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def KResNet152(num_classes=10):
    return KResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


def test():
    net = KResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
