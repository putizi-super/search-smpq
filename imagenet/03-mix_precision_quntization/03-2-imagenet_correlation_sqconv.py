import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import pandas as pd
import numpy as np
import time
from importlib import import_module

import sys
sys.path.append('../../')
from data import imagenet
from data import imagenet_train_val_split
from utils.options import args
import utils.common as utils
from utils.balance import BalancedDataParallel
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

from  search.GA.search_pruning import *
from  search.GA.search_quantization import *

from modules.sq_module.sq_model import *
from modules.sq_module.filter_pruning import *

from tool.meter import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

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

search_loader = imagenet_train_val_split.Data(args)
loader = imagenet.Data(args)

featuremaps = calc_model_featuremap(models.__dict__[args.arch]().to(device),224)
flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch](),input_size=224)

def init_params(model_baseline, model, w_bit, trainLoader):
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
            # s = torch.max(torch.abs(weights.mean() - 2.5 * weights.std()), torch.abs(weights.mean() + 2.5 * weights.std())) # per-tensor
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, verbose=False,threshold=0.01, patience=20)
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
        for batch_idx, (inputs, targets) in enumerate(trainLoader):
            if batch_idx <= 100:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        lr = adjust_learning_rate(optimizer, epoch, batch, len(trainLoader.dataset) // args.train_batch_size)
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
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
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
        # logger.info(
        #     'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
        #         .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        # )
    return accuracy.avg, top5_accuracy.avg

def adjust_learning_rate(optimizer, epoch, batch=None,nBatch=None):
    T_total = args.num_epochs * nBatch
    T_cur = (epoch % args.num_epochs) * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total)) # cosine decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# prepare ids for testing
def test_candidates_model(epoch, candidates_bw_weights, candidates_bw_fm, candidates_prune):
    # test candidates 
    total_candidates = []
    cnt = 0
    for can_bw_weights, can_bw_fm, can_prune in zip(candidates_bw_weights, candidates_bw_fm, candidates_prune):
        model = models.__dict__[args.arch]().to(device)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, checkpoint))
        params = utils.get_params_model(model)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()        

        print('test {}th model'.format(cnt), flush=True)

        can = [[i,j,k] for i,j,k in zip(can_bw_weights,can_bw_fm,can_prune)]

        current_bw_weights = can_bw_weights[:-2]
        current_bw_fm = can_bw_fm[:-2]
        current_prune = can_prune[:-2]
        current_all = [[i,j,k] for i,j,k in zip(current_bw_weights,current_bw_fm,current_prune)]

        print('*** the current (prune,bw_fm,bw_weights) =', end=' ')    
        for x,y,z in current_all:
            print(" ({:2d}%,{:2d},{:2d}) ".format(int(100*z),int(y),int(x)), end=',')
        print("\n")

        print('FLOPs = {:.2f}M ({:.2f}X) '.format(2*can_prune[-1]/1000000, float(total_flops/can_prune[-1])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X) '.format(can_prune[-2]/1000000, float(total_params/can_prune[-2])), flush=True, end=' ')
        print('Avg BW Weights = {:.2f} bit '.format(float(can_bw_weights[-2])/float(total_params)), flush=True, end=' ')
        print('Avg BW FeatureMap= {:.2f} bit '.format(float(can_bw_fm[-2])/float(sum(featuremaps))), flush=True, end=' ')

        t_current_prune = tuple(can_prune[:-2])
        t_current_bw_weights = tuple(can_bw_weights[:-2])
        t_current_bw_fm = tuple(can_bw_fm[:-2])
        search_cnt[0] = 0
        model_sq = convert_to_sqconv(model,t_current_bw_weights,t_current_bw_fm,t_current_prune)
        model_sq = model_sq.to(device)
        # print(model_sq)
        # sys.exit()
        if len(args.gpus) != 1:
            # model = nn.DataParallel(model, device_ids=args.gpus)
            # model = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)), model, dim=0, device_ids=args.gpus)
            model_sq = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)), model_sq, dim=0, device_ids=args.gpus)
            cudnn.benchmark = True

        # print(model_sq)
        model_sq = utils.load_params_model(model_sq,params)

        model_sq = init_params(model,model_sq,t_current_bw_weights,search_loader.trainLoader) # Few Shot Learning
        model_sq, _ = Adaptive_BN(model_sq,search_loader.trainLoader) # Adaptive BN
        top1,top5 = test(model_sq, search_loader.validLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

        print('Score = {:.2f}%'.format(float(top1)), flush=True)
        total_candidates.append([can,float(top1)])
        cnt = cnt + 1
    # pruning
    total_candidates = sorted(total_candidates,key=lambda can: can[-1],reverse=True)
    print('-----------------Epoch {:2d}th result sort---------------'.format(epoch))
    for can in total_candidates:
        print('FLOPs = {:.2f}M ({:.2f}X) |'.format(2*can[0][-1][2]/1000000,float(total_flops/can[0][-1][2])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X)|'.format(can[0][-2][2]/1000000,float(total_params/can[0][-2][2])), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[0][-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[0][-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Score = {:.2f}%'.format(float(can[1])))
    print('---------------------------------------------------------')

    return total_candidates

def select_candidate(global_candidates,interval):
    global_candidates = sorted(global_candidates,key=lambda can: can[-1],reverse=True) # loss reverse=False | Top1 reverse=True
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[::interval,0]
    candidates_top1 = global_candidates[::interval,1]
    # candidates_top5 = global_candidates[::interval,2]
    # candidates_loss = global_candidates[::interval,3]

    print('-----------------Select Gen ---------------')
    for can,top1 in zip(candidates,candidates_top1):
        print('FLOPs = {:.2f}M |'.format(2*can[-1][2]/1000000), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(can[-2][2]/1000000), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Score = {:.2f}% '.format(float(top1)))
    print('-------------------------------------------')

    return candidates,candidates_top1

def main():

    global flops,params,total_params,total_flops,conv_num,fc_num,featuremaps

    print('==> Building model..')
    model = models.__dict__[args.arch]().to(device)

    print(model)

    # if len(args.gpus) != 1:
    #     # model = nn.DataParallel(model, device_ids=args.gpus)
    #     model = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//2), model, dim=args.gpus[0], device_ids=args.gpus)
    #     cudnn.benchmark = True

    checkpoint = torch.load(args.baseline_model, map_location=device)
    model.load_state_dict(utils.convert_keys(model, checkpoint))
    # model_baseline_top1,model_baseline_top5 = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    # print('Model Baseline Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(model_baseline_top1),float(model_baseline_top5)))

    candidates_bw_weights = bw_weights_random_can(args,50,conv_num+fc_num, flops,params,total_flops,total_params)
    candidates_bw_fm = bw_fm_random_can(args,50,conv_num+fc_num, featuremaps,sum(featuremaps))
    candidates_prune = random_can(args, 50, conv_num+fc_num, flops, params, total_flops, total_params)
    total_candidates = test_candidates_model(1,candidates_bw_weights,candidates_bw_fm,candidates_prune)

    global_candidate = total_candidates
    candidates,candidates_top1 = select_candidate(global_candidate,2)
    candidates_bw_weights = [[x for x,_,_ in can] for can in candidates]
    candidates_bw_fm = [[y for _,y,_ in can] for can in candidates]
    candidates_prune = [[z for _,_,z in can] for can in candidates]
    search_top1 = [can for can in candidates_top1]
 
    train_acc = []
    del(model)
    for can_bw_weights, can_bw_fm, can_prune in zip(candidates_bw_weights, candidates_bw_fm, candidates_prune):
        model = models.__dict__[args.arch]().to(device)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, checkpoint))
        params = utils.get_params_model(model)

        can = [[i,j,k] for i,j,k in zip(can_bw_weights,can_bw_fm,can_prune)]

        current_bw_weights = can_bw_weights[:-2]
        current_bw_fm = can_bw_fm[:-2]
        current_prune = can_prune[:-2]
        current_all = [[i,j,k] for i,j,k in zip(current_bw_weights,current_bw_fm,current_prune)]
 
        t_current_prune = tuple(can_prune[:-2])
        t_current_bw_weights = tuple(can_bw_weights[:-2])
        t_current_bw_fm = tuple(can_bw_fm[:-2])
        search_cnt[0] = 0
        model_sq = convert_to_sqconv(model,t_current_bw_weights,t_current_bw_fm,t_current_prune)
        model_sq = model_sq.to(device)
        # print(model_sq)

        if len(args.gpus) != 1:
            # model = nn.DataParallel(model, device_ids=args.gpus)
            model_sq = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)), model_sq, dim=0, device_ids=args.gpus)
            cudnn.benchmark = True

        model_sq = utils.load_params_model(model_sq,params)

        model_sq = init_params(model,model_sq,t_current_bw_weights,loader.trainLoader)
        model_sq, avg_loss = Adaptive_BN(model_sq,loader.trainLoader)

        start_epoch = 0
        best_top1_acc = 0.0
        best_top5_acc = 0.0

        for pname, param in model_sq.named_parameters():
            param.requires_grad = True
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model_sq.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model_sq.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(start_epoch, args.num_epochs):
            train(model_sq, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
            top1,top5 = test(model_sq, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
 
            is_best = best_top1_acc < top1
            if is_best == True:
                best_top1_acc = top1
                best_top5_acc = top5

            print('current model top1 = {:.2f}%  top5 = {:.2f}% '.format(float(top1),float(top5)))
            print('best model top1 = {:.2f}%  top5 = {:.2f}% '.format(float(best_top1_acc),float(best_top5_acc)))
        train_acc.append(float(best_top1_acc))

        del(model_sq)
        del(model) 
 
    print("-----------------------------------")
    print(train_acc)
    print(search_top1)
    x= pd.Series(search_top1)
    y= pd.Series(train_acc)
    kendall = x.corr(y,method="kendall")  
    pearson = x.corr(y,method="pearson")  
    print("search_top1 && train_acc")
    print('kendall = {:.2f}   '.format(float(kendall)))
    print('pearson = {:.2f}   '.format(float(pearson)))

if __name__ == '__main__':
    main()