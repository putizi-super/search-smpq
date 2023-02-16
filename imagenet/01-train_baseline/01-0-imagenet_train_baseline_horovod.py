import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import horovod.torch as hvd

import os
import math
import time
from importlib import import_module

import sys
sys.path.append('../../')
from data import imagenet_horovod
from utils.options import args
from utils.balance import BalancedDataParallel
import utils.common as utils
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

hvd.init()

torch.cuda.set_device(hvd.local_rank())
device = torch.device(f"cuda:{hvd.local_rank()}")
torch.cuda.manual_seed(2021)

cudnn.benchmark = True

# Data
print('==> Preparing data..')
loader = imagenet_horovod.Data(args)

checkpoint = utils.checkpoint(args) 
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))

loss_func = nn.CrossEntropyLoss
if args.mixup > 0.0:
    loss_func = lambda: NLLMultiLabelSmooth(args.label_smoothing)
elif args.label_smoothing > 0.0:
    loss_func = lambda: LabelSmoothing(args.label_smoothing)
loss_func = loss_func()

def train(model, optimizer, loader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    print_freq = len(loader.trainLoader.dataset) // args.train_batch_size // hvd.size() // 10
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader.trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        lr = adjust_learning_rate(optimizer, epoch, batch_idx, len(loader.trainLoader.dataset) // hvd.size() // args.train_batch_size)
        
        output = model(inputs)
        loss = loss_func(output, targets)
        predicted = utils.accuracy(output, targets, topk=topk)

        reduced_loss = reduce_mean(loss, hvd.size())
        reduced_acc1 = reduce_mean(predicted[0], hvd.size())
        reduced_acc5 = reduce_mean(predicted[1], hvd.size())

        losses.update(reduced_loss.item(), inputs.size(0))
        top1.update(reduced_acc1, inputs.size(0))
        top5.update(reduced_acc5, inputs.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % print_freq == 0 and batch_idx != 0 and hvd.rank() == 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'lr {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch_idx * args.train_batch_size * hvd.size(), len(loader.trainLoader.dataset),
                    float(losses.avg),lr, float(top1.avg), float(top5.avg), cost_time
                )
            )
            start_time = current_time

def test(model, loader, topk=(1,)):

    model.eval()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader.testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            predicted = utils.accuracy(outputs, targets, topk=topk)

            reduced_loss = reduce_mean(loss, hvd.size())
            reduced_acc1 = reduce_mean(predicted[0], hvd.size())
            reduced_acc5 = reduce_mean(predicted[1], hvd.size())

            losses.update(reduced_loss.item(), inputs.size(0))
            top1.update(reduced_acc1, inputs.size(0))
            top5.update(reduced_acc5, inputs.size(0))

        if hvd.rank() == 0:
            current_time = time.time()
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(top1.avg), float(top5.avg), (current_time - start_time))
            )
    return top1.avg, top5.avg

def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
    if args.warmup == True:
        if epoch < args.warmup_epochs:
            lr = args.lr * hvd.size() * float(1 + batch + epoch * nBatch) / (5. * nBatch)
        else:
            T_total = args.num_epochs * nBatch
            T_cur = (epoch % args.num_epochs) * nBatch + batch
            lr = 0.5 * args.lr * hvd.size() * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
    else:
        T_total = args.num_epochs * nBatch
        T_cur = (epoch % args.num_epochs) * nBatch + batch
        lr = 0.5 * args.lr * hvd.size() * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    hvd.allreduce(rt, name='barrier')
    rt /= nprocs
    return rt


def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Building model..')
    model = models.__dict__[args.arch]().to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr* hvd.size(),momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr* hvd.size(),weight_decay=args.weight_decay)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(start_epoch, args.num_epochs):
        loader.train_sampler.set_epoch(epoch)
        loader.test_sampler.set_epoch(epoch)
        train(model, optimizer, loader, args, epoch, topk=(1, 5))

        test_top1_acc, test_top5_acc = test(model, loader, topk=(1, 5))

        if hvd.rank() == 0:
            is_best = best_top1_acc < test_top1_acc
            if is_best == True:
                best_top1_acc = test_top1_acc
                best_top5_acc = test_top5_acc

            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            state = {
                'state_dict': model_state_dict,
                'best_top1_acc': best_top1_acc,
                'best_top5_acc': best_top5_acc,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3f}'.format(float(best_top1_acc), float(best_top5_acc)))

if __name__ == '__main__':
    main()