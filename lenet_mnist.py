import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import modules
from modules import Timer
import visdom

cuda_num=0
torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20              # train the training data n times, to save time, we just train 5 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = True   # set to False if you have downloaded


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
test_x = Variable(test_x, volatile=True)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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

class KervNet(nn.Module):
    def __init__(self):
        super(KervNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Kerv2d(
                in_channels=1,              # input height
                out_channels=6,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                mapping='translation',
                kernel_type='polynomial',
                power=nn.Parameter(torch.cuda.FloatTensor([3.8]), requires_grad=True),
                balance=1.7,
                learnable_kernel=True
            ),                              # input shape (1, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (6, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (6, 14, 14)
            nn.Kerv2d(6,16,5,1,0,           # output shape (16, 10, 10)
                mapping='translation',
                kernel_type='linear'),
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

# net = ConvNet()
net = KervNet()

timer = Timer()

if torch.cuda.is_available():
    net.cuda(cuda_num)
    test_x = test_x.cuda(cuda_num)
    test_y = test_y.cuda(cuda_num)

# optimizer = torch.optim.Adam(net.parameters(), lr=LR)
optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9)
loss_func = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9], gamma=0.1)


vis = visdom.Visdom()
train_steps, loss_history, accuracy_history = [],[],[]
power_history, balance_history =[], []
layout_loss = dict(title='Training loss on MNIST', xaxis={'title':'epoch'}, yaxis={'title':'loss'})
layout_accuracy = dict(title='Test accuracy on MNIST', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})
layout_power = dict(title='Power training on MNIST', xaxis={'title':'epoch'}, yaxis={'title':'power'})
layout_balance = dict(title='Balance training on MNIST', xaxis={'title':'epoch'}, yaxis={'title':'balance'})


# training and testing
for epoch in range(EPOCH):
    scheduler.step()
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        if torch.cuda.is_available():
            b_x = b_x.cuda(cuda_num)
            b_y = b_y.cuda(cuda_num)

        net.train()
        output = net(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            net.eval()
            test_output = net(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))

            print('(Epoch:%2d|Step:%4d )' % (epoch, step), '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

            train_steps.append(epoch+step/1200)
            loss_history.append(loss.data[0])
            accuracy_history.append(accuracy)
            power_history.append(net.conv1[0].power.data[0])
            balance_history.append(net.conv1[0].balance.data[0])

            trace_loss = dict(x=train_steps, y=loss_history, mode="markers+lines", type='custom',
                         marker={'color': 'red', 'size': "3"})

            trace_accuracy = dict(x=train_steps, y=accuracy_history, mode="markers+lines", type='custom',
                         marker={'color': 'red', 'size': "3"})
            trace_power = dict(x=train_steps, y=power_history, mode="markers+lines", type='custom',
                         marker={'color': 'red', 'size': "3"})
            trace_balance = dict(x=train_steps, y=balance_history, mode="markers+lines", type='custom',
                         marker={'color': 'red', 'size': "3"})
            vis._send({'data':[trace_loss], 'layout': layout_loss, 'win':'loss'})
            vis._send({'data':[trace_accuracy], 'layout': layout_accuracy, 'win':'accuracy'})
            vis._send({'data':[trace_power], 'layout': layout_power, 'win':'power'})
            vis._send({'data':[trace_balance], 'layout': layout_balance, 'win':'balance'})

torch.save(net.state_dict(), 'checkpoint/kervlenet-poly-mnist.pkl')

# Overall Accuracy
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    vi = Variable(images)
    if torch.cuda.is_available():
        vi = vi.cuda(cuda_num)
        labels = labels.cuda(cuda_num)
    outputs = net(vi)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100.0 * correct / total))
timer.toc()

# print(net.conv1[0].weights)
# print(net.conv1[0].weight)
