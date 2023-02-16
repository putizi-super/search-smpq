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
import random
import numpy as np

sys.path.append('../../')
from utils.options import args
import utils.common as utils
from data import cifar10
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper

import models.cifar10 as models
import tool.pruning as tp
from tool.meter import *
from tool.mergebn import *

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

def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
    #Warmup
    if epoch < 5:
        lr = args.lr * float(1 + batch + epoch * nBatch) / (5. * nBatch)
    else:
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
    # for pname, param in model.named_parameters():
    #     print(pname)
    #     print(param)
    #     # if("a_alpha" in pname) or ("a_beta" in pname) or ("relu" in pname)  or ("bn" in pname):
    #     if ("norm" in pname)  or ("bn" in pname) :
    #         param.requires_grad = True
    #         print(pname)
    #     else:
    #         param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=0.001, weight_decay=1e-5)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, verbose=False, patience=100)
    # criterion_kd = utils.DistributionLoss()
    # loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)

    for batch_idx, (inputs, targets) in enumerate(trainLoader):
        if batch_idx < 300:
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
            if batch <= 300:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg

def filter_prune(model, example_inputs, output_transform):
    model.cpu().eval()
    prunable_module_type = (nn.Conv2d)
    prunable_modules = [ m for m in model.modules() if isinstance(m, prunable_module_type) ]
    pruning_rates = [0.9]*len(prunable_modules)
    DG = tp.DependencyGraph().build_dependency( model, example_inputs=example_inputs, output_transform=output_transform )

    for layer_to_prune, fp_rate in zip(prunable_modules,pruning_rates):
        # select a layer

        # print(layer_to_prune)
        if isinstance( layer_to_prune, nn.Conv2d ):
            prune_fn = tp.prune_conv

        # print(layer_to_prune.weight.detach().cpu().numpy())

        weight = layer_to_prune.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        num_pruned = int(out_channels * fp_rate)
        # print(L1_norm)
        # print(tt)
        prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        # print(prune_index)

        # ch = tp.utils.count_prunable_channels( layer_to_prune )

        # # rand_idx = random.sample( list(range(ch)), min( ch//2, 10 ) )
        # rand_idx = random.sample( list(range(ch)),  ch//2 )
 
        # print(rand_idx)

        plan = DG.get_pruning_plan( layer_to_prune, prune_fn, prune_index)
        # print(plan)
        plan.exec()

    with torch.no_grad():
        out = model( example_inputs )
        if output_transform:
            out = output_transform(out)
    return model

def main():
    start_epoch = 0
    best_acc = 0.0
    input_size = 32
    model = models.__dict__[args.arch]().to(device)
    # print(model)


    if args.baseline == True:
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        baseline_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    params = get_params_mergebn(model)
    model = fillbias(model)
    model = convert_to_mergebn(model)
    model = utils.load_params_model(model,params)
    baseline_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    sys.exit()


    fp_model = filter_prune(copy.deepcopy(model), torch.randn(1, 3, 32, 32), output_transform=None)
    # print(fp_model)

    flops = calc_model_flops(fp_model, input_size)
    parameters = calc_model_parameters(fp_model)
    # print(featuremap)
    # print(sum(flops))
    # print(sum(parameters))

    compression_sparse_flops = sum(flops_baseline) / sum(flops)
    compression_sparse_params = sum(parameters_basleine) / sum(parameters)

    print('Model FLOPs = {:.2f} M | Sparse FLOPs = {:.2f} M | {:.2f} X '.format(sum(flops_baseline)/1000000, sum(flops)/1000000,compression_sparse_flops))
    print('Model Prams = {:.2f} M | Sparse Prams = {:.2f} M | {:.2f} X '.format(sum(parameters_basleine)/1000000, sum(parameters)/1000000,compression_sparse_params))


    # fp_model = init_params(fp_model.to(device),loader.trainLoader)
    fp_model,_ = Adaptive_BN(fp_model.to(device),loader.trainLoader)

    # print(model_fp)
    # baseline_acc = test(model_fp, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    # # convert to sqconv
    # model = convert_to_sqconv(model,w_bit, a_bit, pruning_rate).to(device)
    # model = utils.load_params_model(model,params)

    # flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch](compress_rate=[0.5]*12))

    # featuremap = calc_model_featuremap(model, input_size)
    # flops = calc_model_flops(model, input_size)
    # parameters = calc_model_parameters(model)
    # print(featuremap)
    # print(flops)
    # print(parameters)

    binary_acc = test(fp_model.to(device), loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    binary_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    # print(model)

if __name__ == '__main__':
    main()