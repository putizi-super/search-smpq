import torch
import torch.nn as nn
import torch.optim as optim

import os
import math
import time
from importlib import import_module
from collections import OrderedDict

import sys
sys.path.append('..')
from data import imagenet_dali
from utils.options import args
import utils.common as utils
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
def get_data_set(type='train'):
    if type == 'train':
        print(args.train_batch_size)
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0])
    else:
        print(args.eval_batch_size)
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0])
print("-------------------------------")
trainLoader = get_data_set('train')
print("-------------------------------")
testLoader = get_data_set('test')

featuremaps = calc_model_featuremap(models.__dict__[args.arch]().cuda(),224)
flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch](),input_size=224)

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

    for batch_idx, batch_data in enumerate(trainLoader):
        if batch_idx <= 300:
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
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
        for batch_idx, batch_data in enumerate(trainLoader):
            if batch_idx <= 300:
                inputs = batch_data[0]['data'].to(device)
                targets = batch_data[0]['label'].squeeze().long().to(device)
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
        for batch, batch_data in enumerate(testLoader):
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)

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
    testLoader.reset()
    return accuracy.avg, top5_accuracy.avg

# prepare ids for testing
def test_candidates_model(epoch, search_data, candidates_bw_weights, candidates_bw_fm, candidates_prune):
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
        model_sq = utils.load_params_model(model_sq,params)
        # print(model_sq)

        fsl_loader = trainLoader

        model_sq = init_params(model_sq,fsl_loader) # Few Shot Learning
        model_sq, avg_loss = Adaptive_BN(model_sq,fsl_loader) # Adaptive BN

        top1,top5 = test(model_sq, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        del(model_sq)
        del(model)
        fsl_loader.reset()

        print('Top-1 = {:.2f}% Top-5 = {:.2f}%'.format(float(top1),float(top5)), flush=True)
        # sys.exit()
        total_candidates.append([can,top1,top5])
        cnt = cnt + 1
    # pruning
    total_candidates = sorted(total_candidates,key=lambda can: can[-1],reverse=True)
    print('-----------------Epoch {:2d}th result sort---------------'.format(epoch))
    for can in total_candidates:
        print('FLOPs = {:.2f}M ({:.2f}X) |'.format(2*can[0][-1][2]/1000000,float(total_flops/can[0][-1][2])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X)|'.format(can[0][-2][2]/1000000,float(total_params/can[0][-2][2])), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[0][-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[0][-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(can[1]),float(can[2])))
    print('---------------------------------------------------------')

    return total_candidates

def select_global_candidate(global_candidates,topk):
    global_candidates = sorted(global_candidates,key=lambda can: can[-1],reverse=True)
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[:topk,0]
    candidates_top1 = global_candidates[:topk,1]
    candidates_top5 = global_candidates[:topk,2]

    print('-----------------Global Gen ---------------')
    for can,top1,top5 in zip(candidates,candidates_top1,candidates_top5):
        print('FLOPs = {:.2f}M |'.format(2*can[-1][2]/1000000), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(can[-2][2]/1000000), flush=True, end=' ')
        print('Average BW Weights = {:.2f} bit |'.format(float(can[-2][0])/float(total_params)), flush=True, end=' ')
        print('Average BW FeatureMap = {:.2f} bit |'.format(float(can[-2][1])/float(sum(featuremaps))), flush=True, end=' ')
        print('Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(top1),float(top5)))
    print('-------------------------------------------')

    return candidates,candidates_top1


def main():
    start_epoch = 1
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Load Search Data..')
    search_data = []
    # for batch, batch_data in enumerate(trainLoader):
    #     inputs = batch_data[0]['data'].to(device)
    #     targets = batch_data[0]['label'].squeeze().long().to(device)
    #     if batch <= 300:
    #         search_data.append([inputs,targets])
    #     else:
    #         break

    print('==> Building model..')
    model= models.__dict__[args.arch]().to(device)

    print(model)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
 
    if args.baseline == True:
        model_baseline= models.__dict__[args.arch]().to(device)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(utils.convert_keys(model_baseline, checkpoint))
        # model_baseline_top1,model_baseline_top5 = test(model_baseline, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        # print('Model Baseline Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(model_baseline_top1),float(model_baseline_top5)))

    for epoch in range(start_epoch,  args.search_epochs):
        if epoch == 1:
            checkpoint = torch.load(args.baseline_model, map_location=device)
            model.load_state_dict(utils.convert_keys(model, checkpoint))
            candidates_bw_weights = bw_weights_random_can(args,50,conv_num+fc_num, flops,params,total_flops,total_params)
            candidates_bw_fm = bw_fm_random_can(args,50,conv_num+fc_num, featuremaps,sum(featuremaps))
            candidates_prune = random_can(args, 50, conv_num+fc_num, flops, params, total_flops, total_params)
            total_candidates = test_candidates_model(epoch,search_data,candidates_bw_weights,candidates_bw_fm,candidates_prune)

            global_candidate = total_candidates
            candidates,candidates_acc = select_global_candidate(global_candidate,25)
            candidates_bw_weights = [[x for x,_,_ in can] for can in candidates]
            candidates_bw_fm = [[y for _,y,_ in can] for can in candidates]
            candidates_prune = [[z for _,_,z in can] for can in candidates]
            # total_candidates = np.array(total_candidates)
            # candidates = total_candidates[:,0]
            # candidates_acc = total_candidates[:,1]
        else:
            # strength = adjust_mutation_strength(epoch)
            # mutation = get_mutation(args,candidates,candidates_acc, conv_num+fc_num, 10, 0.1,strength,flops, params, total_flops, total_params)
            bw_weights_mutation = bw_weights_get_mutation(args,epoch,candidates_bw_weights,candidates_acc, conv_num+fc_num, 10, 0.1, 4, 4,flops,params,total_flops,total_params)
            bw_weights_crossover = bw_weights_get_crossover(args,candidates_bw_weights,candidates_acc, conv_num+fc_num, 10,flops,params,total_flops,total_params)
            bw_fm_mutation = bw_fm_get_mutation(args,epoch,candidates_bw_fm,candidates_acc, conv_num+fc_num, 10, 0.1, 4, 4,featuremaps,sum(featuremaps))
            bw_fm_crossover = bw_fm_get_crossover(args,candidates_bw_fm,candidates_acc, conv_num+fc_num, 10,featuremaps,sum(featuremaps))
            prune_mutation = get_mutation(args,epoch,candidates_prune,candidates_acc, conv_num+fc_num, 10, 0.1,4,flops, params, total_flops, total_params)
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
            total_candidates = test_candidates_model(epoch,search_data,candidates_bw_weights,candidates_bw_fm,candidates_prune)
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
        print("best_candidates_bw_weights="+str(best_candidates_bw_weights))
        print("best_candidates_bw_fm="+str(best_candidates_bw_fm))
        print("best_candidates_prune="+str(best_candidates_prune))
        print('FLOPs = {:.2f}M ({:.2f}X) |'.format(2*candidates_prune[0][-1]/1000000,float(total_flops/candidates_prune[0][-1])), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X) |'.format(candidates_prune[0][-2]/1000000,float(total_params/candidates_prune[0][-2])), flush=True, end=' ')
        print('Avg BW Weights = {:.2f} bit |'.format(float(candidates_bw_weights[0][-2])/float(total_params)), flush=True, end=' ')
        print('Avg BW FeatureMap = {:.2f} bit |'.format(float(candidates_bw_fm[0][-2])/float(sum(featuremaps))), flush=True)


if __name__ == '__main__':
    main()