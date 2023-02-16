import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules import loss

import os
import time
from importlib import import_module
import copy
import sys
import math
import numpy as np
sys.path.append('../../')
from utils.options import args
import utils.common as utils
from loss.kd_loss import *
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
criterion_kd = DistributionLoss()
# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

# # # ### Sparse Quantization w=2 a=3 F=5x P=5x  89.31%
# w_bit=[6.0, 4.0, 1.0, 8.0, 2.0, 4.0, 5.0, 2.0, 4.0, 3.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 3.0, 1.0, 8.0]
# a_bit=[7.0, 3.0, 2.0, 3.0, 1.0, 5.0, 3.0, 2.0, 2.0, 3.0, 5.0, 1.0, 2.0, 4.0, 4.0, 2.0, 6.0, 5.0, 3.0, 8.0]
# pruning_rate=[0.23046881831081406, 0.07617270520263154, 0.05877896583296631, 0.23315910514627902, 0.06272026475008476, 0.5134492714071881, 0.2394313563411317, 0.34871476661214584, 0.22604803746712698, 0.18063906759557258, 0.07091614642810852, 0.07259459289482856, 0.3424860876803225, 0.3745949030775168, 0.21345904537184926, 0.08917809567980199, 0.09438664895548153, 0.3582464391326513, 0.1386637600098429, 0.2487389412801343]

# # # ## w=2mp a=5mp # 92.57%
# w_bit=[5.0, 5.0, 3.0, 5.0, 6.0, 4.0, 5.0, 4.0, 3.0, 4.0, 2.0, 3.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 8.0]    
# a_bit=[7.0, 5.0, 5.0, 5.0, 4.0, 7.0, 5.0, 4.0, 3.0, 3.0, 2.0, 6.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 3.0, 8.0]  
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # ## w=2mp a=4mp # 92.52%
# w_bit=[6.0, 2.0, 4.0, 4.0, 5.0, 6.0, 5.0, 6.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]
# a_bit=[8.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 4.0, 5.0, 3.0, 5.0, 5.0, 3.0, 4.0, 6.0, 6.0, 5.0, 5.0, 8.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # ## w=2mp a=4 #  92.31%
# w_bit=[6.0, 3.0, 4.0, 3.0, 4.0, 6.0, 5.0, 4.0, 3.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 4.0]
# a_bit=[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # # # # ## w=2 a=3 # 92.38%
# w_bit=[6.0, 4.0, 6.0, 4.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 5.0]
# a_bit=[8.0, 3.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 7.0, 4.0, 5.0, 8.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # # # # # # # # ### w=2mp a=2mp # 91.40%  
# w_bit=[5.0, 8.0, 3.0, 3.0, 2.0, 8.0, 5.0, 4.0, 3.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 5.0]
# a_bit=[7.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 4.0, 2.0, 7.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# w_bit=[6.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 3.0, 3.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 6.0]  
# a_bit=[5.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 5.0]  
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# w_bit=[6.0, 4.0, 2.0, 1.0, 2.0, 4.0, 6.0, 6.0, 2.0, 4.0, 4.0, 3.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 6.0]
# a_bit=[5.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.0, 3.0, 7.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# w_bit=[6.0, 3.0, 4.0, 2.0, 7.0, 5.0, 3.0, 5.0, 3.0, 4.0, 3.0, 5.0, 4.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 6.0]
# a_bit=[5.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.0, 3.0, 6.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# w_bit=[8]+[3.0]*18+[8]
# a_bit=[8]+[3.0]*18+[8]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

w_bit=[6.0, 5.0, 5.0, 4.0, 3.0, 5.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 6.0]
a_bit=[6.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 6.0]
pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def train(model_teacher, model, optimizer, trainLoader, args, epoch, topk=(1,)):
    model_teacher.eval()
    model.train()

    # scheduler.step()
    # for param_group in optimizer.param_groups:
    #     cur_lr = param_group['lr']
    # print('learning_rate:', cur_lr)

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    # alpha1 = utils.step(epoch,args.num_epochs)
    alpha1 = utils.exp(epoch,args.num_epochs)
    # alpha1 = utils.log(epoch,args.num_epochs)
    # alpha1 = utils.linear(epoch,args.num_epochs)

    for batch, (inputs, targets) in enumerate(trainLoader):
        lr = adjust_learning_rate(optimizer, epoch, batch, len(trainLoader.dataset) // args.train_batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        output_teacher = model_teacher(inputs)
        loss_ce = loss_func(output, targets)
        loss_kd = criterion_kd(output, output_teacher)
        # loss = alpha1*loss_ce+(1.0-alpha1)*loss_kd
        # loss = 0.5*loss_ce + 0.5*loss_kd
        loss = loss_ce
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

def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
    T_total = args.num_epochs * nBatch
    T_cur = (epoch % args.num_epochs) * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
#     T_total = args.num_epochs * nBatch
#     T_cur = (epoch % args.num_epochs) * nBatch + batch
#     lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
#     for i in range(len(optimizer.param_groups)):
#         param_group = optimizer.param_groups[i]
#         if i == 0:
#             param_group['lr'] = lr
#         else:
#             param_group['lr'] = lr*0.05
#     return lr

def init_params(model_baseline, model, trainLoader):
    model_baseline.eval()
    ## weights step size init
    cnt_conv = 0
    w_s = []
    for m in model_baseline.modules():
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            # print(m.weight.data.size())
            weights = copy.deepcopy(m.weight.data)
            # s = torch.max(torch.abs(weights.abs().mean(dim=list(range(1, weights.dim())), keepdim=True) - 3.0 * weights.abs().std(dim=list(range(1, weights.dim())), keepdim=True)), torch.abs(weights.abs().mean(dim=list(range(1, weights.dim())), keepdim=True) + 3.0 * weights.abs().std(dim=list(range(1, weights.dim())), keepdim=True)))  / (2 ** (args.w_bit - 1)) # per-channel
            s = torch.max(torch.abs(weights.mean() - 3.0 * weights.std()), torch.abs(weights.mean() + 3.0 * weights.std()))  / (2 ** (w_bit[cnt_conv]- 1)) # per-tensor
            # s = torch.max(torch.abs(weights.mean() - 3.0 * weights.std()), torch.abs(weights.mean() + 3.0 * weights.std())) # per-tensor
            # s =  (weights.max()-weights.min()) / (2 ** w_bit[cnt_conv]-1) # per-tensor
 
            # s = weights.abs().mean() * 2 / (2 ** (args.w_bit - 1) - 1 ** 0.5) # per-tensor
            # s = weights.abs().mean(dim=list(range(1, weights.dim())), keepdim=True) * 2 / (2 ** (args.w_bit - 1) - 1 ** 0.5) # per-channel
            w_s.append(s)
            cnt_conv = cnt_conv + 1

    cnt_conv = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            m.w_s.data = w_s[cnt_conv]
            cnt_conv = cnt_conv + 1

    model.train()

    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            for pname, param in m.named_parameters():
                param.requires_grad = True
        elif(isinstance(m, Scale_HardTanh)) or (isinstance(m, Scale_PACT)):
            for pname, param in m.named_parameters():
                param.requires_grad = True
        elif(isinstance(m, SQConv)) or (isinstance(m, SQLinear)):
            for pname, param in m.named_parameters():
                if ("w_s" in pname) or ("a_s" in pname) or ("a_beta" in pname):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
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
    start_epoch = 1
    best_acc = 0.0

    model_baseline = models.__dict__[args.arch]().to(device)
    model_teacher = models.__dict__[args.arch_teacher]().to(device)

    # print(model_baseline)

    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location=device)
        ckpt_teacher = torch.load(args.teacher_model, map_location=device)
        model_baseline.load_state_dict(ckpt['state_dict'])
        model_teacher.load_state_dict(ckpt_teacher['state_dict'])
        baseline_acc = test(model_baseline, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        params = utils.get_params_model(model_baseline)

    if device == 'cuda':
        model_baseline = torch.nn.DataParallel(model_baseline)
        cudnn.benchmark = True

    for p in model_baseline.parameters():
        p.requires_grad = False
    model_baseline.eval()

    # convert to sqconv
    model = convert_to_sqconv(copy.deepcopy(model_baseline),w_bit, a_bit, pruning_rate).to(device)

    print(model)

    # calc featuremap & params & flops
    layer_featuremap = calc_model_featuremap(model,32)
    quatization_featuremap =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_featuremap,a_bit))))
    compression_featuremap = sum(layer_featuremap)*32/quatization_featuremap

    layer_params = calc_model_parameters(model)
    quatization_params =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_params,w_bit))))
    compression_quantization_params = sum(layer_params)*32/quatization_params
 
    sparse_params =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_params,pruning_rate))))
    compression_sparse_params = sum(layer_params)/sparse_params

    layer_flops = calc_model_flops(model, 32, mul_add=False)
    sparse_flops =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_flops,pruning_rate))))
    compression_sparse_flops = sum(layer_flops)/sparse_flops

    print('Model FLOPs      = {:.2f} M         | Sparse       FLOPs      = {:.2f} M         | {:.2f} X '.format(sum(layer_flops)/1000000, sparse_flops/1000000,compression_sparse_flops))
    print('Model Prams      = {:.2f} M (num)   | Sparse       Prams      = {:.2f} M  (num)  | {:.2f} X '.format(sum(layer_params)/1000000, sparse_params/1000000,compression_sparse_params))
    print('Model Prams      = {:.2f} M (Byte)  | Quantization Prams      = {:.2f} M  (Byte) | AvgBit = {:.2f} bit | {:.2f} X | '.format(sum(layer_params*4)/1000000, quatization_params/8/1000000,32/compression_quantization_params,compression_quantization_params))
    print('Model FeatureMap = {:.2f} M (Byte)  | Quantization FeatureMap = {:.2f} M  (Byte) | AvgBit = {:.2f} bit | {:.2f} X | '.format(sum(layer_featuremap*4)/1000000, quatization_featuremap/8/1000000,32/compression_featuremap,compression_featuremap))

    if args.baseline == True:
        model = utils.load_params_model(model,params)
        # add baseline -> Few shot Learning (1000 batchs) for quantization params
        model = init_params(copy.deepcopy(model_baseline), model,loader.trainLoader)
        model, avg_loss = Adaptive_BN(model,loader.trainLoader)
    else:
        ## scrach ->  Adaptive-BN
        # model = init_params(model,model,loader.trainLoader)
        model,avg_loss = Adaptive_BN(model,loader.trainLoader)

    binary_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        model_teacher = torch.nn.DataParallel(model_teacher)
        cudnn.benchmark = True

    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    for pname, param in model.named_parameters():
        param.requires_grad = True

    # all_parameters = filter(lambda p: p.requires_grad == True, model.parameters())

    # weight_parameters = []
    # for pname, p in model.named_parameters():
    #     if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
    #         weight_parameters.append(p)
    # weight_parameters_id = list(map(id, weight_parameters))
    # other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #             [{'params' : other_parameters},
    #             {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #             lr=args.lr,)
    # elif args.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(
    #             [{'params' : other_parameters},
    #             {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #             lr=args.lr,)      

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.num_epochs), last_epoch=-1)


    # quan_parameters = []

    # for m in model.modules():
    #     if (isinstance(m, nn.Conv2d)):
    #         for pname, param in m.named_parameters():
    #             if 'a_s' in pname or 'a_beta' in pname or 'w_s' in pname:
    #                 quan_parameters.append(param)
    #     elif (isinstance(m, nn.Linear)):
    #         for pname, param in m.named_parameters():
    #             if 'a_s' in pname or 'a_beta' in pname or 'w_s' in pname:
    #                 quan_parameters.append(param)

    # quan_parameters_id = list(map(id, quan_parameters))
    # other_parameters = list(filter(lambda p: id(p) not in quan_parameters_id, all_parameters))

    # if args.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(
    #             [{'params' : other_parameters},
    #              {'params' : quan_parameters, 'weight_decay' : 0,'lr' : args.lr*0.05}], lr=args.lr, weight_decay=args.weight_decay,momentum=args.momentum)
    # elif args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #             [{'params' : other_parameters},
    #              {'params' : quan_parameters, 'weight_decay' : 0,'lr' : args.lr*0.05}], lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.num_epochs+1):

        train(model_teacher, model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

        # cnt_conv = 0
        # b1 = []
        # b2 = []
        # k1 = []
        # k2 = []
        # for m in model.modules():
        #     # if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        #     if (isinstance(m, Scale_HardTanh)):
        #         b1.append(m.b1.data.tolist()[0])
        #         b2.append(m.b2.data.tolist()[0])
        #         k1.append(m.k1.data.tolist()[0])
        #         k2.append(m.k2.data.tolist()[0])
        #         cnt_conv+=1
        # print(k1)        
        # print(b1)
        # print(k2)        
        # print(b2)        

        # cnt_conv = 0
        # b1 = []
        # b2 = []
        # k1 = []
        # k2 = []
        # for m in model.modules():
        #     # if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        #     if (isinstance(m, Scale_HardTanh)):
        #         b1.append(m.b1)
        #         b2.append(m.b2)
        #         k1.append(m.k1)
        #         k2.append(m.k2.data.tolist()[0])
        #         cnt_conv+=1
        # print(b1)
        # print(b2)        
        # print(k1)        
        # print(k2)        

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
