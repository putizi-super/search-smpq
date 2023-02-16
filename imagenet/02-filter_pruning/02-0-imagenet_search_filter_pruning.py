import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import time
from importlib import import_module
from collections import OrderedDict

import sys
sys.path.append('../../')
from data import imagenet_train_val_split
from utils.options import args
import utils.common as utils
from utils.balance import BalancedDataParallel
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

from  search.GA.search_filter_pruning_new import *
from  search.GA.search_quantization import *

from modules.sq_module.sq_model import *
from modules.sq_module.filter_pruning import *

from tool.meter import *
import tool.pruning as tp

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

loader = imagenet_train_val_split.Data(args)

featuremaps = calc_model_featuremap(models.__dict__[args.arch](),224)
# flops,params,total_params,total_flops,conv_num,fc_num = get_params_flops(model=models.__dict__[args.arch]())
layer_params = calc_model_parameters(models.__dict__[args.arch]())
layer_flops = calc_model_flops(models.__dict__[args.arch](), 224, mul_add=False)
total_params = sum(layer_params)
total_flops = sum(layer_flops)
print(total_params)
print(total_flops)

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
        for batch_idx, (inputs, targets) in enumerate(trainLoader):
            if batch_idx <= 100:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg


def filter_prune(baseline_model, example_inputs, output_transform, pruning_rates, method='Att'):
    baseline_model.cpu().eval()
    prunable_module_type = (nn.Conv2d)
    prunable_modules = [ m for m in baseline_model.modules() if isinstance(m, prunable_module_type) ]
    DG = tp.DependencyGraph().build_dependency(baseline_model, example_inputs=example_inputs, output_transform=output_transform )

    for layer_to_prune, fp_rate in zip(prunable_modules,pruning_rates):
        # select a layer

        # print(layer_to_prune)
        if isinstance( layer_to_prune, nn.Conv2d ):
            prune_fn = tp.prune_conv

        weight = layer_to_prune.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        num_pruned = int(out_channels * fp_rate)
        # print(num_pruned)

        if method == 'l1':
            # # # L1 norm
            L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
            prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        elif method == 'l2':
            # # # L2 norm
            L2_norm = np.sqrt(np.sum(np.power(weight, 2), axis=(1,2,3)))
            prune_index = np.argsort(L2_norm)[:num_pruned].tolist()
        elif method == 'GM':
            # # # GM
            geo_dist = [np.sum(np.power((wi - wj), 2) for wj in weight) for wi in weight]
            prune_index = np.argsort(geo_dist)[:num_pruned].tolist()
        elif method == 'Att':
            # # # Att
            Att_weights = np.sqrt(np.sum(np.power(weight, 2), axis=0))/weight.shape[0]
            Att_gap = np.abs(weight) - Att_weights
            L2_norm = np.sqrt(np.sum(np.power(Att_gap, 2), axis=(1,2,3)))
            prune_index = np.argsort(L2_norm)[:num_pruned].tolist()

        plan = DG.get_pruning_plan( layer_to_prune, prune_fn, prune_index)
        plan.exec()

    with torch.no_grad():
        out = baseline_model( example_inputs )
        if output_transform:
            out = output_transform(out)
    return baseline_model

def test(model, testLoader, topk=(1,)):
    model.eval()

    loss = utils.AverageMeter()
    accurseacy = utils.AverageMeter()
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

# prepare ids for testing
def test_candidates_model(epoch, candidates_prune_fp):
    baseline_model = models.__dict__[args.arch]()
    ckpt = torch.load(args.baseline_model, map_location=device)
    origin_state_dict = utils.convert_keys(baseline_model, ckpt)  #？
    baseline_model.load_state_dict(origin_state_dict)

    # test candidates
    total_candidates = []
    cnt = 0
    for can_prune_fp in candidates_prune_fp:
        print('test {}th model'.format(cnt), flush=True)

        print('*** the current (Filter Prune Rate) =', end=' ')  
        for x in can_prune_fp[:-2]:
            print(" {:2d}% ".format(int(100*x)), end=',')
        print("\n")

        print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}%) '.format(can_prune_fp[-1]/1000000, float(total_flops/can_prune_fp[-1]), float(1.0-can_prune_fp[-1]/total_flops)*100), flush=True, end=' ')
        print('Filter Pruning PARAMs = {:.2f}M ({:.2f}X | {:.2f}% ) '.format(can_prune_fp[-2]/1000000, float(total_params/can_prune_fp[-2]), float(1.0-can_prune_fp[-2]/total_params)*100), flush=True, end=' ')

        t_current_prune_fp = can_prune_fp[:-2]

        # get filter_prune
        model_fp = filter_prune(copy.deepcopy(baseline_model),torch.randn(1, 3, 224, 224),output_transform=None,pruning_rates=t_current_prune_fp)
        model_fp = model_fp.to(device)

        if len(args.gpus) != 1:
            # model = nn.DataParallel(model, device_ids=args.gpus)
            model_fp = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//1.3), model_fp, dim=args.gpus[0], device_ids=args.gpus)
            cudnn.benchmark = True
        # train
        model_fp = init_params(model_fp,loader.trainLoader) # Few Shot Learning
        model_fp, avg_loss = Adaptive_BN(model_fp,loader.trainLoader) # Adaptive BN
        # test
        top1,top5 = test(model_fp, loader.validLoader, topk=( 1, 5) if args.data_set == 'imagenet' else (1, ))

        print('Score  = {:.2f}%'.format(float(top1)), flush=True)
        # sys.exit()
        total_candidates.append([can_prune_fp,float(top1)])

        cnt = cnt + 1
    # pruning
    total_candidates = sorted(total_candidates,key=lambda can: can[-1],reverse=True)
    print('-----------------Epoch {:2d}th result sort---------------'.format(epoch))
    for can in total_candidates:
        print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}%) |'.format(can[0][-1]/1000000,float(total_flops/can[0][-1]),float(1.0-can[0][-1]/total_flops)*100), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% )|'.format(can[0][-2]/1000000,float(total_params/can[0][-2]),float(1.0-can[0][-2]/total_params)*100), flush=True, end=' ')
        print('Score = {:.2f}%   '.format(float(can[1])))
    print('---------------------------------------------------------')

    return total_candidates


def select_global_candidate(global_candidates,topk):
    # topk = 25
    global_candidates = sorted(global_candidates,key=lambda can: can[-1],reverse=True) # loss reverse=False | Top1 reverse=True
    global_candidates = np.array(global_candidates)
    candidates = global_candidates[:topk,0]
    candidates_top1 = global_candidates[:topk,1]
    # candidates_top5 = global_candidates[:topk,2]

    print('-----------------Global Gen ---------------')
    for can,top1 in zip(candidates,candidates_top1):
        print('FLOPs = {:.2f}M |'.format(float(can[-1]/1000000)), flush=True, end=' ')
        print('PARAMs = {:.2f}M |'.format(float(can[-2]/1000000)), flush=True, end=' ')
        print('Score = {:.4f}'.format(float(top1)))
    print('-------------------------------------------')

    return candidates,candidates_top1

def main():
    start_epoch = 1
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Building model..')
    model= models.__dict__[args.arch]().to(device)

    print(model)
 
    if args.baseline == True:
        model_baseline= models.__dict__[args.arch]().to(device)
        ckpt = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(utils.convert_keys(model_baseline, ckpt))
        # model_baseline_top1,model_baseline_top5 = test(model_baseline, loader.testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        # print('Model Baseline Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(model_baseline_top1),float(model_baseline_top5)))

    prunable_module_type = (nn.Conv2d)
    prunable_modules = [ m for m in model_baseline.modules() if isinstance(m, prunable_module_type) ]
    args.FP_Rate_Len = len(prunable_modules)

    for epoch in range(start_epoch,  args.search_epochs):
        if epoch == 1:
            candidates_prune_fp = random_can_fp(args, 224, 50, args.FP_Rate_Len, total_flops, total_params)
            total_candidates = test_candidates_model( epoch, candidates_prune_fp)
            global_candidate = total_candidates
            candidates,candidates_top1 = select_global_candidate(global_candidate,25)

        else:
            # 进化            
            prune_mutation_fp = get_mutation_fp(args, 224, epoch,candidates,candidates_top1, args.FP_Rate_Len, 10, 0.1,4, total_flops, total_params)
            # 交叉
            prune_crossover_fp = get_crossover_fp(args, 224, candidates,candidates_top1, args.FP_Rate_Len, 10, total_flops, total_params)
            candidates_prune_fp = []
            candidates_prune_fp.extend(prune_mutation_fp)
            candidates_prune_fp.extend(prune_crossover_fp)
            # 获得多组 candidates_prune_fp进行测试
            total_candidates = test_candidates_model( epoch, candidates_prune_fp)
            # sparse_rate = quantization_model_search(epoch,total_candidates[0][0])
            # print(type(global_candidate))
            for can in total_candidates:
                global_candidate.insert(0, can)
            # global_candidate 二维列表 ：candidates = global_candidates[:topk,0] candidates_top1 = global_candidates[:topk,1] candidates_top5 = global_candidates[:topk,2]
            candidates,candidates_top1 = select_global_candidate(global_candidate,25)

        # summary
        best_candidates_prune = [can for can in candidates[0][:-2]]
        print("best_candidates_prune="+str(best_candidates_prune))
        print('Filter Pruning FLOPs = {:.2f}M ({:.2f}X | {:.2f}% ) |'.format(candidates[0][-1]/1000000,float(total_flops/candidates[0][-1]),float(1.0-candidates[0][-1]/total_flops)*100), flush=True, end=' ')
        print('PARAMs = {:.2f}M ({:.2f}X | {:.2f}% ) |'.format(candidates[0][-2]/1000000,float(total_params/candidates[0][-2]),float(1.0-candidates[0][-2]/total_params)*100), flush=True, end=' ')
        print('Score = {:.2f}   '.format(float(candidates_top1[0])))

if __name__ == '__main__':
    main()
