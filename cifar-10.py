'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.kresnet import *
from models.kgooglenet import *
from torch.autograd import Variable

cuda_num = 0
epoch_num = 200
milestones = [50, 100, 150]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log', type=str, help='log folder')
parser.add_argument('--folder', default='./results/', type=str, help='checkpoint saved folder')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

folder = args.folder

if not os.path.isdir(folder+args.log):
    os.mkdir(folder+args.log)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(folder+args.log), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(folder+args.log+'/'+args.log+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
else:
    f = open(folder+args.log+'/'+args.log+'.txt',"a+")
    f.write("epoch |  train_loss | test_loss | train_acc | test_acc | best_acc\n")
    f.write('milestones:'+str(milestones)+'\n')
    f.close()

    print('==> Building model..')
    print("epoch |  train_loss | test_loss | train_acc | test_acc | best_acc")

    # net = VGG('VGG19')
    # net = ResNet18()
    # net = ResNet34()
    # net = ResNet50()
    # net = ResNet101()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

    # net = KResNet18()
    # net = KResNet34()
    # net = KResNet50()
    # net = KResNet101()
    # net = KResNet152()
    net = KGoogLeNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


if use_cuda:
    net.cuda(cuda_num)
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(cuda_num), targets.cuda(cuda_num)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    return (train_loss/(batch_idx+1), 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(cuda_num), targets.cuda(cuda_num)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'criterion': criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
        if not os.path.isdir(folder+args.log):
            os.mkdir(folder+args.log)
        torch.save(state, folder+args.log+'/'+args.log+'.t7')
        best_acc = acc
    return (test_loss/(batch_idx+1),acc)


for epoch in range(start_epoch, start_epoch+epoch_num):
    scheduler.step()
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    f = open(folder+args.log+'/'+args.log+'.txt',"a+")
    f.write("%3d %f %f %f %f %f\n" % (epoch, train_loss, test_loss, train_acc, test_acc, best_acc))
    f.close()
    print("%3d %f %f %f %f %f\n" % (epoch, train_loss, test_loss, train_acc, test_acc, best_acc))

