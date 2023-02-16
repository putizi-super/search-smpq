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

sys.path.append('../../')
from utils.options import args
from utils.summary import *
import utils.common as utils

from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper

from data import cifar10_train_val_split
import models.cifar10 as models

from  search.GA.search_pruning import *
from  search.GA.search_quantization import *

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

loader = cifar10_train_val_split.Data(args)

model = models.__dict__[args.arch]().to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

featuremaps = calc_model_featuremap(models.__dict__[args.arch]().cuda(),32)
flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch]())

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

        current_time = time.time()
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

        # loader = cifar10_train_val_split.Data(args)
        # fls = loader.trainLoader
        model_sq = init_params(model, model_sq, t_current_bw_weights, loader.trainLoader) # Few Shot Learning
        model_sq,_ = Adaptive_BN(model_sq,loader.trainLoader) # Adaptive BN
        top1 = test(model_sq, loader.vaildLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

        print('Score = {:.2f}%'.format(float(top1)), flush=True)
        # del(fls)
        # sys.exit()
        total_candidates.append([can,top1])
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

def select_global_candidate(global_candidates,topk):
    global_candidates = sorted(global_candidates,key=lambda can: can[-1],reverse=True)
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[:topk,0]
    candidates_acc = global_candidates[:topk,1]

    print('-----------------Global Gen ---------------')
    for can,acc in zip(candidates,candidates_acc):
        print('FLOPs = {:.2f}M |'.format(2*can[-1][2]/1000000), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(can[-2][2]/1000000), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Score = {:.2f}%'.format(float(acc)))
    print('-------------------------------------------')

    return candidates, candidates_acc

def main():
    start_epoch = 1
    best_acc = 0.0
    # global model

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
            candidates,candidates_acc = select_global_candidate(global_candidate,25)
            candidates_bw_weights = [[x for x,_,_ in can] for can in candidates]
            candidates_bw_fm = [[y for _,y,_ in can] for can in candidates]
            candidates_prune = [[z for _,_,z in can] for can in candidates]
            # total_candidates = np.array(total_candidates)
            # candidates = total_candidates[:,0]
            # candidates_loss = total_candidates[:,1]
        else:
            # strength = adjust_mutation_strength(epoch)
            # mutation = get_mutation(args,candidates,candidates_loss, conv_num+fc_num, 10, 0.1,strength,flops, params, total_flops, total_params)
            bw_weights_mutation = bw_weights_get_mutation(args,epoch,candidates_bw_weights,candidates_acc, conv_num+fc_num, 10, 0.25, 4, 4,flops,params,total_flops,total_params)
            bw_weights_crossover = bw_weights_get_crossover(args,candidates_bw_weights,candidates_acc, conv_num+fc_num, 10,flops,params,total_flops,total_params)
            bw_fm_mutation = bw_fm_get_mutation(args,epoch,candidates_bw_fm,candidates_acc, conv_num+fc_num, 10, 0.25, 4, 4,featuremaps,sum(featuremaps))
            bw_fm_crossover = bw_fm_get_crossover(args,candidates_bw_fm,candidates_acc, conv_num+fc_num, 10,featuremaps,sum(featuremaps))
            prune_mutation = get_mutation(args,epoch,candidates_prune,candidates_acc, conv_num+fc_num, 10, 0.25, 4,flops, params, total_flops, total_params)
            prune_crossover = get_crossover(args,candidates_prune,candidates_acc, conv_num+fc_num, 10,flops, params, total_flops, total_params)
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
            candidates,candidates_acc = select_global_candidate(global_candidate,25)
            candidates_bw_weights = [[x for x,_,_ in can] for can in candidates]
            candidates_bw_fm = [[y for _,y,_ in can] for can in candidates]
            candidates_prune = [[z for _,_,z in can] for can in candidates]

        # summary
        best_candidates_bw_weights = candidates_bw_weights[0][:-2]
        best_candidates_bw_fm = candidates_bw_fm[0][:-2]
        best_candidates_prune = candidates_prune[0][:-2]
        print("w_bit="+str(best_candidates_bw_weights))
        print("a_bit="+str(best_candidates_bw_fm))
        print("pruning_rate="+str(best_candidates_prune))
        print('FLOPs = {:.2f}M ({:.2f}X) |'.format(2*candidates_prune[0][-1]/1000000,float(total_flops/candidates_prune[0][-1])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X) |'.format(candidates_prune[0][-2]/1000000,float(total_params/candidates_prune[0][-2])), flush=True, end=' ')
        print('Avg BW Weights = {:.2f} bit |'.format(float(candidates_bw_weights[0][-2])/float(total_params)), flush=True, end=' ')
        print('Avg BW FeatureMap = {:.2f} bit |'.format(float(candidates_bw_fm[0][-2])/float(sum(featuremaps))), flush=True)

if __name__ == '__main__':
    main()