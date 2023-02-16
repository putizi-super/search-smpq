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
import pandas as pd
import numpy as np

sys.path.append('../../../')
from utils.options import args
import utils.common as utils
from data import cifar10
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper

from data import cifar10
from data import cifar10_train_val_split

from  search.GA.search_filter_pruning import *
from  search.GA.search_quantization import *
# from  modules.search_sq.pruning import *
# from  modules.search_sq.quantization import *

from modules.sq_module.sq_model import *
from modules.sq_module.filter_pruning import *

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
loader_search = cifar10_train_val_split.Data(args)
loader = cifar10.Data(args)

model = models.__dict__[args.arch]().to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

featuremaps = calc_model_featuremap(models.__dict__[args.arch]().cuda(),32)
flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch]())

def init_params(model, trainLoader):
    model.train()
    for pname, param in model.named_parameters():
        # print(pname)
        # if("a_alpha" in pname) or ("a_beta" in pname) or ("relu" in pname)  or ("bn" in pname):
        if ("norm" in pname)  or ("bn" in pname) :
            param.requires_grad = True
            # print(pname)
        else:
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

# prepare ids for testing
def test_candidates_model(epoch, candidates_prune_fp):
    # test candidates 
    total_candidates = []
    cnt = 0
    for can_prune_fp in candidates_prune_fp:
        model = models.__dict__[args.arch]().to(device)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, checkpoint))
        params = utils.get_params_model(model)

        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        
        # print('Top-1 = {:.2f}%'.format(top1), flush=True)

        print('test {}th model'.format(cnt), flush=True)

        print('*** the current (Filter Prune Rate) =', end=' ')  
        for x in can_prune_fp[:-2]:
            print(" {:2d}% ".format(int(100*x)), end=',')
        print("\n")


        print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X) '.format(2*can_prune_fp[-1]/1000000, float(total_flops/can_prune_fp[-1])), flush=True, end=' ')
        print('Filter Pruning PARAMs = {:.2f}M ({:.2f}X) '.format(can_prune_fp[-2]/1000000, float(total_params/can_prune_fp[-2])), flush=True, end=' ')

        t_current_prune_fp = can_prune_fp[:-2]
 
        model_fp = models.__dict__[args.arch](t_current_prune_fp).to(device)
        # model_fp = utils.load_params_model_fp_resize(model_fp,params)
        model_fp = load_params_model_fp(model_fp,params)
 
        model_fp = init_params(model_fp,loader_search.trainLoader) # Few Shot Learning
        model_fp, _ = Adaptive_BN(model_fp,loader_search.trainLoader) # Adaptive BN
        top1 = test(model_fp, loader_search.vaildLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        del(model_fp)
        del(model) 

        print('Score = {:.2f}% '.format(float(top1)), flush=True)
        total_candidates.append([can_prune_fp,float(top1)])
        cnt = cnt + 1
    # pruning
    total_candidates = sorted(total_candidates,key=lambda can: can[-1],reverse=True)
    print('-----------------Epoch {:2d}th result sort---------------'.format(epoch))
    for can in total_candidates:
        print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X) |'.format(can[0][-1]/1000000,float(total_flops/can[0][-1])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X)|'.format(can[0][-2]/1000000,float(total_params/can[0][-2])), flush=True, end=' ')
        print('Score = {:.2f}%'.format(float(can[1])))
    print('---------------------------------------------------------')

    return total_candidates

def select_candidate(global_candidates,interval):
    global_candidates = sorted(global_candidates,key=lambda can: can[-1],reverse=True)
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[::interval,0]
    candidates_acc = global_candidates[::interval,1]

    print('-----------------Select Gen ---------------')
    for can,acc in zip(candidates,candidates_acc):
        print('FLOPs = {:.2f}M |'.format(2*can[-1]/1000000), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(can[-2]/1000000), flush=True, end=' ')
        print('Score = {:.2f}%'.format(float(acc)))
    print('-------------------------------------------')

    return candidates,candidates_acc

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
            accuracy.update(predicted[0].item(), inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1].item(), inputs.size(0))

        # current_time = time.time()
        # if len(topk) == 1:
        #     logger.info(
        #         'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
        #         .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
        #     )
        # else:
        #     logger.info(
        #         'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
        #             .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        #     )
    if len(topk) == 1:
        return accuracy.avg
    else:
        return top5_accuracy.avg

def adjust_learning_rate(optimizer, epoch, batch=None, nBatch=None):
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

def main():

    global model
    global flops,params,total_params,total_flops,conv_num,fc_num,featuremaps

    ckpt = torch.load(args.baseline_model, map_location=device)
    model.load_state_dict(utils.convert_keys(model, ckpt))
    baseline_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    print('baseline model top1 accuracy = {:.2f}%   '.format(float(baseline_acc)))

    candidates_prune_fp = random_can_fp(args, 32, 50, args.FP_Rate_Len, total_flops, total_params)


    total_candidates = test_candidates_model(1, candidates_prune_fp)

    candidates,candidates_acc = select_candidate(total_candidates,5)
 
    search_acc = [can for can in candidates_acc]
 
    train_acc = []
    del(model)
    for can_prune_fp in  candidates:
        model = models.__dict__[args.arch]().to(device)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, checkpoint))
        params = utils.get_params_model(model)

        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        t_current_prune_fp = can_prune_fp[:-2]
 
        model_fp = models.__dict__[args.arch](t_current_prune_fp).to(device)
        # model_fp = utils.load_params_model_fp_resize(model_fp,params)
        model_fp = utils.load_params_model_fp(model_fp,params)
 
        model_fp = init_params(model_fp,loader.trainLoader)
        model_fp, avg_loss = Adaptive_BN(model_fp,loader.trainLoader)

        start_epoch = 0
        best_acc = 0.0

        for pname, param in model_fp.named_parameters():
            param.requires_grad = True
 
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model_fp.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model_fp.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(start_epoch, args.num_epochs):
            train(model_fp, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
            test_acc = test(model_fp, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
            best_acc = max(best_acc, test_acc)
            print('current model top1 accuracy = {:.2f}%   '.format(float(test_acc)))
            print('baseline model top1 accuracy = {:.2f}%   '.format(float(best_acc)))
        train_acc.append(best_acc)

        del(model_fp)
        del(model) 
        
    print(train_acc)
    print(search_acc)
    x= pd.Series(search_acc)
    y= pd.Series(train_acc)
    kendall = x.corr(y,method="kendall")  
    pearson = x.corr(y,method="pearson")  
    
    print('kendall = {:.2f}   '.format(float(kendall)))
    print('pearson = {:.2f}   '.format(float(pearson)))


if __name__ == '__main__':
    main()
