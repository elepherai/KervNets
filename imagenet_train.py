import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import time
import visdom
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import MultiStepLR
from utils import convert_secs2time, time_string, time_file_str
from collections import OrderedDict
import imagenet_models

model_names = sorted(name for name in imagenet_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(imagenet_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',default='dev/remote-imagenet/',
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='record/', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='kresnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: kresnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

args = parser.parse_args()
args.prefix = time_file_str()

milestones = [30, 60, 90]

# visualization
vis = visdom.Visdom()
train_steps, test_steps, loss_history, top1_history, top5_history = [],[],[],[],[]
test_top1_history, test_top5_history = [],[]
crop10_top1_history, crop10_top5_history = [],[]
layout_loss = dict(title='Training loss on ImageNet', xaxis={'title':'epoch'}, yaxis={'title':'loss'})
layout_top1 = dict(title='Top 1 accuracy on ImageNet', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})
layout_top5 = dict(title='Top 5 accuracy on ImageNet', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})
layout_test_top1 = dict(title='Test Top 1 accuracy on ImageNet', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})
layout_test_top5 = dict(title='Test Top 5 accuracy on ImageNet', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})
layout_10crop_top1 = dict(title='10 Cropping Top 1 accuracy', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})
layout_10crop_top5 = dict(title='10 Cropping Top 5 accuracy', xaxis={'title':'epoch'}, yaxis={'title':'accuracy'})

def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    # log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch,args.prefix)), 'a')

    # create model
    print("=> creating model '{}'\n".format(args.arch))
    model = imagenet_models.__dict__[args.arch](1000)
    print("=> Model : {}\n".format(args.arch))
    # print("=> Model : {}\n".format(model))
    # print("=> parameter : {}\n".format(args))

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            del checkpoint['state_dict']['module.conv1.weights']    # fix this bug
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val-pytorch')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader_10 = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])),
    batch_size=16, shuffle=False,
    num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    best_p1, best_p5, best_c10_p1, best_c10_p5 = 0, 0, 0, 0
    filename = os.path.join(args.save_dir, 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{}.{}.pth.tar'.format(args.arch, args.prefix))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(args.arch, epoch, args.epochs, time_string(), need_time))

        scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, epoch)
        c10_p1, c10_p5 = validate_10crop(val_loader_10, model, criterion, epoch)

        # store the best
        best_p1 = prec1 if prec1 > best_p1 else best_p1
        best_p5 = prec5 if prec5 > best_p5 else best_p5
        best_c10_p1 = c10_p1 if c10_p1 > best_c10_p1 else best_c10_p1
        best_c10_p5 = c10_p5 if c10_p5 > best_c10_p5 else best_c10_p5

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename, bestname)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    
    # loss_history, top1_history, top5_history
    print('training loss', loss_history)
    print('training top1 accuracy', top1_history)
    print('training top5 accuracy', top5_history)
    print('testing top1 accuracy', test_top1_history)
    print('testing top5 accuracy', test_top5_history)
    print('10-croping top1', crop10_top1_history)
    print('10-croping top5', crop10_top5_history)
    print('Best single-crop top 1: {0:.2f}\n'
    'Best single-crop top 5: {1:.2f}\n'
    'Best 10-crop top 1: {2:.2f}\n'
    'Best 10-crop top 5: {3:.2f}\n'.format(best_p1, best_p5, best_c10_p1, best_c10_p5))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_steps.append(epoch+i/len(train_loader))
            loss_history.append(loss.data[0])
            top1_history.append(top1.avg)
            top5_history.append(top5.avg)

            trace_loss = dict(x=train_steps, y=loss_history, mode="markers+lines", type='custom',
                marker={'color': 'red', 'size': "3"})
            trace_top1 = dict(x=train_steps, y=top1_history, mode="markers+lines", type='custom',
                         marker={'color': 'red', 'size': "3"})
            trace_top5 = dict(x=train_steps, y=top5_history, mode="markers+lines", type='custom',
                         marker={'color': 'red', 'size': "3"})

            vis._send({'data':[trace_loss], 'layout': layout_loss, 'win':'loss'})
            vis._send({'data':[trace_top1], 'layout': layout_top1, 'win':'top1'})
            vis._send({'data':[trace_top5], 'layout': layout_top5, 'win':'top5'})
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    test_steps.append(epoch+1)
    test_top1_history.append(top1.avg)
    test_top5_history.append(top5.avg)
    trace_test_top1 = dict(x=test_steps, y=test_top1_history, mode="markers+lines", type='custom',
                    marker={'color': 'red', 'size': "3"})
    trace_test_top5 = dict(x=test_steps, y=test_top5_history, mode="markers+lines", type='custom',
                    marker={'color': 'red', 'size': "3"})
    vis._send({'data':[trace_test_top1], 'layout': layout_test_top1, 'win':'Test top1'})
    vis._send({'data':[trace_test_top5], 'layout': layout_test_top5, 'win':'Test top5'})
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

    return top1.avg,top5.avg


def validate_10crop(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        bs, ncrops, c, h, w = input_var.size()
        temp_output = model(input_var.view(-1, c, h, w))
        output = temp_output.view(bs, ncrops, -1).mean(1)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('10cropping-Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    test_steps.append(epoch+1)
    crop10_top1_history.append(top1.avg)
    crop10_top5_history.append(top5.avg)
    trace_crop_top1 = dict(x=test_steps, y=crop10_top1_history, mode="markers+lines", type='custom',
                    marker={'color': 'red', 'size': "3"})
    trace_crop_top5 = dict(x=test_steps, y=crop10_top5_history, mode="markers+lines", type='custom',
                    marker={'color': 'red', 'size': "3"})
    vis._send({'data':[trace_crop_top1], 'layout': layout_10crop_top1, 'win':'10 crop top1'})
    vis._send({'data':[trace_crop_top5], 'layout': layout_10crop_top5, 'win':'10 crop top5'})
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))

    return top1.avg,top5.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr
    decay_lr = [30, 45, 60]
    for item in decay_lr:
        if item == epoch:
            lr = args.lr * (0.1 ** (decay_lr.index(item)+1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
