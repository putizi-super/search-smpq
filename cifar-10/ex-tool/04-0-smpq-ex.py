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

sys.path.append('..')
from utils.options import args
import utils.common as utils

from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper

from data import cifar10
import models.cifar10 as models

from  search.GA.search_pruning import *
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

loader = cifar10.Data(args)

model = models.__dict__[args.arch]().to(device)


if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

featuremaps = calc_model_featuremap(models.__dict__[args.arch]().cuda(),32)
flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch]())

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

        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        
        # print('Top-1 = {:.2f}%'.format(top1), flush=True)

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
        model_sq = utils.load_params_model(model_sq,params)


        fsl_loader = loader.trainLoader

        model_sq = init_params(model_sq,fsl_loader) # Few Shot Learning
        model_sq, avg_loss = Adaptive_BN(model_sq,fsl_loader) # Adaptive BN
        top1 = test(model_sq, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        del(model_sq)
        del(model) 
        del(fsl_loader) 

        print('Top-1 = {:.2f}%  Loss = {:.4f}'.format(float(top1),float(avg_loss)), flush=True)     
        # sys.exit()
        total_candidates.append([can,top1,avg_loss])
        cnt = cnt + 1
    # pruning
    total_candidates = sorted(total_candidates,key=lambda can: can[-1],reverse=False)
    print('-----------------Epoch {:2d}th result sort---------------'.format(epoch))
    for can in total_candidates:
        print('FLOPs = {:.2f}M ({:.2f}X) |'.format(2*can[0][-1][2]/1000000,float(total_flops/can[0][-1][2])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X)|'.format(can[0][-2][2]/1000000,float(total_params/can[0][-2][2])), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[0][-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[0][-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Top1 = {:.2f}% | Loss = {:.4f} '.format(float(can[1]),float(can[2])))
    print('---------------------------------------------------------')

    return total_candidates

def select_global_candidate(global_candidates,topk):
    global_candidates = sorted(global_candidates,key=lambda can: can[-1],reverse=False)
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[:topk,0]
    candidates_acc = global_candidates[:topk,1]
    candidates_loss = global_candidates[:topk,2]

    print('-----------------Global Gen ---------------')
    for can,acc,loss in zip(candidates,candidates_acc,candidates_loss):
        print('FLOPs = {:.2f}M |'.format(2*can[-1][2]/1000000), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(can[-2][2]/1000000), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Top1 = {:.2f}% | Loss = {:.4f} '.format(float(acc),float(loss)))
    print('-------------------------------------------')

    return candidates,candidates_acc,candidates_loss

def main():
    start_epoch = 1
    best_acc = 0.0
    global params
    global model

    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, ckpt))
        baseline_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        print('baseline model top1 accuracy = {:.2f}%   '.format(float(baseline_acc)))

    for epoch in range(start_epoch,  args.search_epochs):
        if epoch == 1:
            ckpt = torch.load(args.baseline_model, map_location=device)
            model.load_state_dict(utils.convert_keys(model, ckpt))
            candidates_bw_weights = bw_weights_random_can(args,50,conv_num+fc_num, flops,params,total_flops,total_params)
            candidates_bw_fm = bw_fm_random_can(args,50,conv_num+fc_num, featuremaps,sum(featuremaps))
            candidates_prune = random_can(args, 50, conv_num+fc_num, flops, params, total_flops, total_params)
            total_candidates = test_candidates_model(epoch,candidates_bw_weights,candidates_bw_fm,candidates_prune)

            global_candidate = total_candidates
            candidates,candidates_acc,candidates_loss = select_global_candidate(global_candidate,25)
            candidates_bw_weights = [[x for x,_,_ in can] for can in candidates]
            candidates_bw_fm = [[y for _,y,_ in can] for can in candidates]
            candidates_prune = [[z for _,_,z in can] for can in candidates]
            # total_candidates = np.array(total_candidates)
            # candidates = total_candidates[:,0]
            # candidates_loss = total_candidates[:,1]
        else:
            # strength = adjust_mutation_strength(epoch)
            # mutation = get_mutation(args,candidates,candidates_loss, conv_num+fc_num, 10, 0.1,strength,flops, params, total_flops, total_params)
            bw_weights_mutation = bw_weights_get_mutation(args,epoch,candidates_bw_weights,candidates_loss, conv_num+fc_num, 10, 0.1, 4, 4,flops,params,total_flops,total_params)
            bw_weights_crossover = bw_weights_get_crossover(args,candidates_bw_weights,candidates_loss, conv_num+fc_num, 10,flops,params,total_flops,total_params)
            bw_fm_mutation = bw_fm_get_mutation(args,epoch,candidates_bw_fm,candidates_loss, conv_num+fc_num, 10, 0.1, 4, 4,featuremaps,sum(featuremaps))
            bw_fm_crossover = bw_fm_get_crossover(args,candidates_bw_fm,candidates_loss, conv_num+fc_num, 10,featuremaps,sum(featuremaps))
            prune_mutation = get_mutation(args,epoch,candidates_prune,candidates_loss, conv_num+fc_num, 10, 0.1,4,flops, params, total_flops, total_params)
            prune_crossover = get_crossover(args,candidates_prune,candidates_loss, conv_num+fc_num, 10,flops, params, total_flops, total_params)
            candidates_bw_weights = []
            candidates_bw_fm = []
            candidates_prune = []
            candidates_bw_weights.extend(bw_weights_mutation)
            candidates_bw_weights.extend(bw_weights_crossover)
            candidates_bw_fm.extend(bw_fm_mutation)
            candidates_bw_fm.extend(bw_fm_crossover)
            candidates_prune.extend(prune_mutation)
            candidates_prune.extend(prune_crossover)
            total_candidates = test_candidates_model(epoch,candidates_bw_weights,candidates_bw_fm,candidates_prune)
            # sparse_rate = quantization_model_search(epoch,total_candidates[0][0])
            # print(type(global_candidate))
            for can in total_candidates:
                global_candidate.insert(0, can)
            candidates,candidates_acc,candidates_loss = select_global_candidate(global_candidate,25)
            candidates_bw_weights = [[x for x,_,_ in can] for can in candidates]
            candidates_bw_fm = [[y for _,y,_ in can] for can in candidates]
            candidates_prune = [[z for _,_,z in can] for can in candidates]

        # summary
        best_candidates_bw_weights = candidates_bw_weights[0][:-2]
        best_candidates_bw_fm = candidates_bw_fm[0][:-2]
        best_candidates_prune = candidates_prune[0][:-2]
        print("best_candidates_bw_weights="+str(best_candidates_bw_weights))
        print("best_candidates_bw_fm="+str(best_candidates_bw_fm))
        print("best_candidates_prune="+str(best_candidates_prune))
        print('FLOPs = {:.2f}M ({:.2f}X) |'.format(2*candidates_prune[0][-1]/1000000,float(total_flops/candidates_prune[0][-1])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X) |'.format(candidates_prune[0][-2]/1000000,float(total_params/candidates_prune[0][-2])), flush=True, end=' ')
        print('Avg BW Weights = {:.2f} bit |'.format(float(candidates_bw_weights[0][-2])/float(total_params)), flush=True, end=' ')
        print('Avg BW FeatureMap = {:.2f} bit |'.format(float(candidates_bw_fm[0][-2])/float(sum(featuremaps))), flush=True)

    #finetuning
    start_epoch = 0
    best_acc = 0.0

    model_baseline = models.__dict__[args.arch]().to(device)

    # print(model_baseline)

    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(ckpt['state_dict'])
        baseline_acc = test(model_baseline, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        params = utils.get_params_model(model_baseline)
    
    w_bit = tuple(best_candidates_bw_weights)
    a_bit = tuple(best_candidates_bw_fm)
    pruning_rate = tuple(best_candidates_prune)

    # convert to sqconv
    search_cnt[0] = 0
    model = convert_to_sqconv(model_baseline,w_bit, a_bit, pruning_rate).to(device)

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

    # print('Model FLOPs      = {:.2f} M         | Sparse       FLOPs      = {:.2f} M         | {:.2f} X '.format(sum(layer_flops)/1000000, sparse_flops/1000000,compression_sparse_flops))
    # print('Model Prams      = {:.2f} M (num)   | Sparse       Prams      = {:.2f} M  (num)  | {:.2f} X '.format(sum(layer_params)/1000000, sparse_params/1000000,compression_sparse_params))
    # print('Model Prams      = {:.2f} M (Byte)  | Quantization Prams      = {:.2f} M  (Byte) | AvgBit = {:.2f} bit | {:.2f} X | '.format(sum(layer_params*4)/1000000, quatization_params/8/1000000,32/compression_quantization_params,compression_quantization_params))
    # print('Model FeatureMap = {:.2f} M (Byte)  | Quantization FeatureMap = {:.2f} M  (Byte) | AvgBit = {:.2f} bit | {:.2f} X | '.format(sum(layer_featuremap*4)/1000000, quatization_featuremap/8/1000000,32/compression_featuremap,compression_featuremap))

    if args.baseline == True:
        model = utils.load_params_model(model,params)
        # # add baseline -> Few shot Learning (1000 batchs) for quantization params
        model = init_params(model,loader.trainLoader)
        model, avg_loss = Adaptive_BN(model,loader.trainLoader)
        print(avg_loss)
    else:
        ## scrach ->  Adaptive-BN
        model,avg_loss = Adaptive_BN(model,loader.trainLoader)

    binary_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    for pname, param in model.named_parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.num_epochs):

        train(model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

        # cnt_conv = 0
        # a_s = []
        # w_s = []
        # for m in model.modules():
        #     if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        #         a_s.append(m.a_s.data.tolist())
        #         w_s.append(m.w_s.data.tolist())
        #         cnt_conv+=1
        # print(a_s)
        # print(w_s)        

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        print('Current Top1 = {:.2f}% '.format(float(test_acc)))
        print('Best Top1 = {:.2f}% '.format(float(best_acc)))

        model_state_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }

        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))
 
    print('Sparse FLOPs = {:.2f} M | {:.2f} X | Sparse Prams = {:.2f} M (num) | {:.2f} X | AvgBit(w) = {:.2f} bit | {:.2f} X | AvgBit(FeatureMap) = {:.2f} bit | {:.2f} X |search_acc = {:.3f}% | search_loss = {:.4f} | fineturn_acc = {:.3f}%'\
        .format(sparse_flops/1000000,compression_sparse_flops,sparse_params/1000000,compression_sparse_params,32/compression_quantization_params,compression_quantization_params,32/compression_featuremap,compression_featuremap,candidates_acc[0],avg_loss,float(best_acc)))

if __name__ == '__main__':
    main()