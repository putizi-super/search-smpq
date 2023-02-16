import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import time
from importlib import import_module
import copy
import sys
import math
import numpy as np
sys.path.append('../../../')
from utils.options import args
import utils.common as utils
from data import cifar10
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
from modules.sq_module.sq_model import *
from modules.sq_module.filter_pruning import *
import models.cifar10 as models

from tool.meter import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss
if args.mixup > 0.0:
    loss_func = lambda: NLLMultiLabelSmooth(args.label_smoothing)
elif args.label_smoothing > 0.0:
    loss_func = lambda: LabelSmoothing(args.label_smoothing)
loss_func = loss_func()
# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

best_candidates_prune=[0.5497675621132571, 0.8915426972044601, 0.6474568200421598, 0.27802945332987056, 0.30687324383533937, 0.47659291824369177, 0.24473208733165133, 0.24184820959167838, 0.16524532339250475, 0.23028017326043065, 0.3998431061957427, 0.25134253766953407, 0.3230766513446486, 0.27796447337795893, 0.12282758030459368, 0.43120768877092325, 0.1399160750803982, 0.2752121204034469, 0.1469748158920574, 0.3001065960326845, 0.34309626196970894, 0.0857398241723984, 0.17399516209713878, 0.2714274051715634, 0.5195917908282545, 0.18355012413803432, 0.2269593205747219, 0.41363255832304774, 0.0014657530281777758, 0.39938660406933413]

# best_candidates_prune=[1.0]+[1.0]*29

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):
        lr = adjust_learning_rate(optimizer, epoch, batch, len(trainLoader.dataset) // args.train_batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'lr {:.4f}\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),lr,
                        float(losses.avg), float(accuracy.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'lr {:.4f}\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),lr,
                        float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accuracy.avg
    else:
        return top5_accuracy.avg

# def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
#     #Warmup
#     if epoch < 5:
#         lr = args.lr * float(1 + batch + epoch * nBatch) / (5. * nBatch)
#     else:
# 	    T_total = args.num_epochs * nBatch
# 	    T_cur = (epoch % args.num_epochs) * nBatch + batch
# 	    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
    T_total = args.num_epochs * nBatch
    T_cur = (epoch % args.num_epochs) * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def init_params(model, trainLoader):
    model.train()

    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            for pname, param in m.named_parameters():
                param.requires_grad = True
        else:
            for pname, param in m.named_parameters():
                param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=0.01, weight_decay=1e-5)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, verbose=False, patience=20)
    # criterion_kd = utils.DistributionLoss()
    # loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)

    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        if batch_idx < 100:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            total_loss = loss_func(output, targets)
            # print(total_loss)
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
        else:
            break
    return model

def Adaptive_BN(model, trainLoader):
    losses = utils.AverageMeter()
    model.train()
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(trainLoader):
            if batch <= 100:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg

def main():
    start_epoch = 0
    best_acc = 0.0

    model_baseline = models.__dict__[args.arch]().to(device)

    layer_params = calc_model_parameters(model_baseline)
    layer_flops = calc_model_flops(model_baseline, 32, mul_add=False)

    print(model_baseline)

    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(ckpt['state_dict'])
        baseline_acc = test(model_baseline, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        params = utils.get_params_model(model_baseline)
        state_dict_baseline = copy.deepcopy(model_baseline.state_dict())

    # convert to sqconv
    model = models.__dict__[args.arch](best_candidates_prune).to(device)

    print(model)

    # calc params & flops
 
    sparse_params = calc_model_parameters(model)
    compression_sparse_params = sum(layer_params)/sum(sparse_params)

    sparse_flops = calc_model_flops(model, 32, mul_add=False)
    compression_sparse_flops = sum(layer_flops)/sum(sparse_flops)

    print('Model FLOPs = {:.2f} M | Sparse FLOPs = {:.2f} M | {:.2f} X '.format(sum(layer_flops)/1000000, sum(sparse_flops)/1000000,compression_sparse_flops))
    print('Model Prams = {:.2f} M | Sparse Prams = {:.2f} M | {:.2f} X '.format(sum(layer_params)/1000000, sum(sparse_params)/1000000,compression_sparse_params))

    if args.baseline == True:
        # model = load_params_model_fp(model,params)
        # model = load_params_pruned_resnet_l2(model, state_dict_baseline)
        model = load_params_pruned_resnet(model, state_dict_baseline)
        # # add baseline -> Few shot Learning (1000 batchs) for quantization params
        model = init_params(model,loader.trainLoader)
        model, avg_loss = Adaptive_BN(model,loader.trainLoader)
        # print("avg_loss:", avg_loss)

    binary_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    for pname, param in model.named_parameters():
        param.requires_grad = True

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))     

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }

        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()