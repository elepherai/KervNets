import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import kervolution
from kervolution import Timer
import math
# import visdom

import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epoch', default=20, type=int, help='epoch')
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
parser.add_argument('--model', default='lekervnet', type=str, help='model')
parser.add_argument('--kernel_type', default='polynomial', type=str, help='model')
parser.add_argument('--learnable_kernel', default=True, type=bool, help='model')
parser.add_argument('--verbose', default=False, type=bool, help='log verbose [only first 4 epoch]')
parser.add_argument('--log', type=str, help='log folder')
parser.add_argument('--folder', default='./results/', type=str, help='checkpoint saved folder')
parser.add_argument('--cuda_num', default=0, type=int, help='cuda num')

args = parser.parse_args()

folder = args.folder
if not os.path.isdir(folder+args.log):
    os.mkdir(folder+args.log)

cuda_num=args.cuda_num
torch.cuda.set_device(cuda_num)
torch.manual_seed(1)    # reproducible
# Hyper Parameters
EPOCH = args.epoch              # train the training data n times, to save time, we just train 5 epoch
BATCH_SIZE = 50
DOWNLOAD_MNIST = True   # set to False if you have downloaded
best_acc = 0
LR = args.lr              # learning rate

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Data.DataLoader(dataset=test_data, batch_size=2000, shuffle=False)
test_x, test_y = test_loader.__iter__().next()
test_x = Variable(test_x)


class KervNet(nn.Module):    
    def __init__(self):
        super(KervNet, self).__init__()
          
        self.features = nn.Sequential(
            nn.Kerv2d(1, 32, kernel_size=3, stride=1, padding=1, kernel_type=args.kernel_type, learnable_kernel=args.learnable_kernel),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Kerv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Kerv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Kerv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
          
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )
          
        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LeConvNet(nn.Module):
    def __init__(self):
        super(LeConvNet, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=6,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (6, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (6, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (6, 14, 14)
            nn.Conv2d(6, 16, 5, 1, 0),      # output shape (16, 10, 10)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (16, 5, 5)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

class LeKervNet(nn.Module):
    def __init__(self):
        super(LeKervNet, self).__init__()
        self.kerv1 = nn.Sequential(
            nn.Kerv2d(
                in_channels=1,              # input height
                out_channels=6,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                kernel_type = args.kernel_type,
                learnable_kernel = args.learnable_kernel,
            ),                              # input shape (1, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (6, 14, 14)
        )
        self.kerv2 = nn.Sequential(         # input shape (6, 14, 14)
            nn.Kerv2d(6,16,5,1,0,           # output shape (16, 10, 10)
                mapping='translation',
                kernel_type='linear'
                ),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (16, 5, 5)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.kerv1(x)
        x = self.kerv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

if args.model == 'leconvnet':
    net = LeConvNet()
elif args.model == 'lekervnet':
    net = LeKervNet()
elif args.model == 'kervnet':
    net = KervNet()


timer = Timer()

if torch.cuda.is_available():
    net.cuda(cuda_num)
    test_x = test_x.cuda(cuda_num)
    test_y = test_y.cuda(cuda_num)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)


timer = Timer()
time_use = 0

def train(epoch):
    global timer, time_use
    timer.start()
    scheduler.step()
    train_loss = 0
    correct = 0
    total = 0
    time_use += timer.end()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        timer.start()
        inputs, targets = inputs.cuda(cuda_num), targets.cuda(cuda_num)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        time_use += timer.end()

        if epoch < 4 and batch_idx%100==0 and args.verbose:
            test_loss, test_acc = test()
            f = open(folder+args.log+'/'+args.log+'_verbose.txt',"a+")
            f.write("%d %d %f %f %f\n" % (epoch+1, batch_idx, test_loss, test_acc, time_use))
            f.close()

    return (train_loss/(batch_idx+1), 100.*correct/total)


def test():
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(cuda_num), targets.cuda(cuda_num)
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = loss_func(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return (test_loss/(batch_idx+1),acc)

f = open(folder+args.log+'/'+args.log+'.txt',"a+")
f.write("epoch | train_loss | test_loss | train_acc | test_acc | best_acc | time_use\n")
f.close()

print("epoch, train_loss, test_loss, train_acc, test_acc, best_acc, time_use")

for epoch in range(EPOCH):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    f = open(folder+args.log+'/'+args.log+'.txt',"a+")
    f.write("%d %f %f %f %f %f %f\n" % (epoch+1, train_loss, test_loss, train_acc, test_acc, best_acc, time_use))
    f.close()
    print("%3d %f %f %f %f %f %f" % (epoch+1, train_loss, test_loss, train_acc, test_acc, best_acc, time_use))
