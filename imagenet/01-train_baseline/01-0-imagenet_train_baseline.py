import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import time
from importlib import import_module

import sys
sys.path.append('../../')
from data import imagenet_dali
from data import imagenet
from utils.options import args
from utils.balance import BalancedDataParallel
import utils.common as utils
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

# device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss
if args.mixup > 0.0:
    loss_func = lambda: NLLMultiLabelSmooth(args.label_smoothing)
elif args.label_smoothing > 0.0:
    loss_func = lambda: LabelSmoothing(args.label_smoothing)
loss_func = loss_func()

# Data
print('==> Preparing data..')
def get_data_set(type='train'):
    if type == 'train':
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,num_threads=8, crop=224, device_id=args.gpus[0])
    else:
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size, num_threads=8, crop=224, device_id=args.gpus[0])
if args.dali==True:        
    trainLoader = get_data_set('train')
    testLoader = get_data_set('test')
else:
    loader = imagenet.Data(args)
    trainLoader = loader.trainLoader
    testLoader = loader.testLoader  

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):
    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    start_time = time.time()
    if args.dali == True:
        print_freq = trainLoader._size // args.train_batch_size // 10
        for batch, batch_data in enumerate(trainLoader):
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)

            lr = adjust_learning_rate(optimizer, epoch, batch, trainLoader._size // args.train_batch_size)

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            losses.update(loss.item(), inputs.size(0))
            optimizer.step()

            prec1 = utils.accuracy(output, targets, topk=topk)
            accuracy.update(prec1[0], inputs.size(0))
            top5_accuracy.update(prec1[1], inputs.size(0))

            if batch % print_freq == 0 and batch != 0:
                current_time = time.time()
                cost_time = current_time - start_time
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'lr {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, trainLoader._size,
                        float(losses.avg),lr, float(accuracy.avg), float(top5_accuracy.avg), cost_time
                    )
                )
                start_time = current_time
        trainLoader.reset()
    else:
        print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
        for batch, (inputs, targets) in enumerate(trainLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            lr = adjust_learning_rate(optimizer, epoch, batch,  len(trainLoader.dataset) // args.train_batch_size)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            losses.update(loss.item(), inputs.size(0))
            optimizer.step()

            prec1 = utils.accuracy(output, targets, topk=topk)
            accuracy.update(prec1[0], inputs.size(0))
            top5_accuracy.update(prec1[1], inputs.size(0))

            if batch % print_freq == 0 and batch != 0:
                current_time = time.time()
                cost_time = current_time - start_time
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'lr {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size,  len(trainLoader.dataset),
                        float(losses.avg),lr, float(accuracy.avg), float(top5_accuracy.avg), cost_time
                    )
                )
                start_time = current_time


def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    if args.dali == True:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(testLoader):
                inputs = batch_data[0]['data'].to(device)
                targets = batch_data[0]['label'].squeeze().long().to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                predicted = utils.accuracy(outputs, targets, topk=topk)
                accuracy.update(predicted[0], inputs.size(0))
                top5_accuracy.update(predicted[1], inputs.size(0))

            current_time = time.time()
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
        testLoader.reset()
    else:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                predicted = utils.accuracy(outputs, targets, topk=topk)
                accuracy.update(predicted[0], inputs.size(0))
                top5_accuracy.update(predicted[1], inputs.size(0))

            current_time = time.time()
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    return accuracy.avg, top5_accuracy.avg

def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
    if args.warmup == True:
        if epoch < args.warmup_epochs:
            lr = args.lr * float(1 + batch + epoch * nBatch) / (5. * nBatch)
        else:
            T_total = args.num_epochs * nBatch
            T_cur = (epoch % args.num_epochs) * nBatch + batch
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
    else:
        T_total = args.num_epochs * nBatch
        T_cur = (epoch % args.num_epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Building model..')
    model = models.__dict__[args.arch]().to(device)

    print(model)

    if len(args.gpus) != 1:
        # model = nn.DataParallel(model, device_ids=args.gpus, dim=0)
        if args.dali == True:
            model = BalancedDataParallel(int(args.train_batch_size//len(args.gpus))//len(args.gpus)*2, model, dim=0, device_ids=args.gpus).to(device)
        else:
            model = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)), model, dim=0, device_ids=args.gpus).to(device)
        # model = BalancedDataParallel(0, model, dim=args.gpus[0]).to(device)
        cudnn.benchmark = True

    # all_parameters = model.parameters()
    # weight_parameters = []
    # for pname, p in model.named_parameters():
    #     if p.ndimension() == 4 or 'conv' in pname:
    #         weight_parameters.append(p)
    # weight_parameters_id = list(map(id, weight_parameters))
    # other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    # if args.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(
    #             [{'params' : other_parameters},
    #             {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #             lr=args.lr,momentum=args.momentum)
    # elif args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #             [{'params' : other_parameters},
    #             {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #             lr=args.lr)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.num_epochs):

        train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))

        test_top1_acc, test_top5_acc = test(model, testLoader, topk=(1, 5))

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