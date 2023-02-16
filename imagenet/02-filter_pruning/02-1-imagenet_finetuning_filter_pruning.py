import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import time
import numpy as np
from importlib import import_module

import sys
sys.path.append('../../')
from data import imagenet_dali
from utils.options import args
from utils.balance import BalancedDataParallel
import utils.common as utils
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

from modules.sq_module.sq_model import *
from modules.sq_module.filter_pruning import *

from tool.meter import *
import tool.pruning as tp

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
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0])
    else:
        return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                   num_threads=4, crop=224, device_id=args.gpus[0])
trainLoader = get_data_set('train')
testLoader = get_data_set('test')

# # # ### ResNet18 f=40.0% p=25.44%   Top1= % Top5= %
# best_candidates_prune=[0.0765423583475218, 0.0431397873705765, 0.3604879948980855, 0.07570219706804253, 0.003955508586705909, 0.35812285579711467, 0.027665122041667, 0.1050930463153434, 0.3851931652569454, 0.171250996945271, 0.11066086440486701, 0.030212829747693182, 0.0778367133421459, 0.04429476082505848, 0.13335514079497054, 0.24877259003534105, 0.028767109690543448, 0.11799433937581072, 0.1580157451045507, 0.09653686507729031] 
# best_candidates_prune=[0.13000531131786164, 0.23017675339963184, 0.21622255597420387, 0.1438758043111763, 0.2479143772493188, 0.19128012450378057, 0.07847775898755469, 0.09425893090544693, 0.18797398336918775, 0.09169998897440683, 0.21932742840053532, 0.08710552627961932, 0.025571730204287185, 0.19762691606154187, 0.09193461421094544, 0.11427719191803482, 0.004804020267285634, 0.09851583781990689, 0.11650882196908312, 0.02005632779158789]
# best_candidates_prune=[0.1244096531311836, 0.226208680358899, 0.2164309168627822, 0.16044102984570258, 0.24805113267442588, 0.1374307142733439, 0.07134327977475087, 0.0944997950573233, 0.2506740007019793, 0.10417193574832757, 0.2076227798584719, 0.08749819329686902, 0.007446137243496921, 0.195469815090776, 0.0985161216198269, 0.09492347135708074, 0.00460450661727225, 0.09631742533757862, 0.11093706702004896, 0.029685415913782542]
best_candidates_prune=[0.13284260710389922, 0.2479362180061966, 0.28338048097615653, 0.14343180914837103, 0.16455207190718016, 0.306270123249856, 0.15957261895551816, 0.10964740080644612, 0.13725018681458512, 0.07360346663084107, 0.11236062012729925, 0.11290507161943161, 0.036777353504201145, 0.21684501141835388, 0.10894620907968927, 0.05736561916379897, 0.0304780176030202, 0.04313926736454888, 0.13346925709674362, 0.001919037133510715]

# best_candidates_prune=[0.0]*1+[0.2]*19

# # # # # ### ResNet50  f= p=   Top1= % Top5= % 
 
# # # # # # ### MobileNet_v1 f=50.1% p=28.5%   Top1= % Top5= % 
best_candidates_prune=[0.14557351480973707, 0.30596939518112415, 0.2300369736069741, 0.35755165635417396, 0.11676625558939531, 0.06849898374692148, 0.20327312265193562, 0.18160187810871292, 0.18961698035617697, 0.12483174155315106, 0.2216663015863248, 0.2022807595726384, 0.2858786710472807, 0.07705749385633784, 0.14387346752330765, 0.3445533121609278, 0.214959140999345, 0.2632975491124254, 0.16024764870171146, 0.13797421569921645, 0.25446975770899527, 0.151212564734987, 0.11799078903420993, 0.07140189886973525, 0.029855014496595285, 0.09653450376114386, 0.004392834753880721]

# # # # ### MobileNet_v2 f= p=   Top1= % Top5= % 
 

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = trainLoader._size // args.train_batch_size // 10
    start_time = time.time()
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

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
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
    return accuracy.avg, top5_accuracy.avg

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

    for batch_idx, batch_data in enumerate(trainLoader):
        if batch_idx <= 100:
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
            if batch_idx <= 100:
                inputs = batch_data[0]['data'].to(device)
                targets = batch_data[0]['label'].squeeze().long().to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg

def filter_prune(baseline_model, example_inputs, output_transform, pruning_rates, method='l1'):
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
            L1_norm = np.sum(np.abs(weight), axis=(1,2,3))
            prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        elif method == 'l2':
            # # # L2 norm
            L2_norm = np.sqrt(np.sum(np.power(weight, 2), axis=(1,2,3)))
            prune_index = np.argsort(L2_norm)[:num_pruned].tolist()
        elif method == 'GM':
            # # # GM
            geo_dist = [np.sum(np.power((wi - wj), 2) for wj in weight) for wi in weight]
            prune_index = np.argsort(geo_dist)[:num_pruned].tolist()

        plan = DG.get_pruning_plan( layer_to_prune, prune_fn, prune_index)
        plan.exec()

    with torch.no_grad():
        out = baseline_model( example_inputs )
        if output_transform:
            out = output_transform(out)
    return baseline_model

def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Building model..')
    model_baseline = models.__dict__[args.arch]().to(device)

    print(model_baseline)

    layer_params = calc_model_parameters(model_baseline)
    layer_flops = calc_model_flops(model_baseline, 224, mul_add=False)

    if args.baseline == True:
        model_baseline= models.__dict__[args.arch]().to(device)
        ckpt = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(utils.convert_keys(model_baseline, ckpt))
        # model_baseline_top1,model_baseline_top5 = test(model_baseline, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        # print('Model Baseline Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(model_baseline_top1),float(model_baseline_top5)))

    # convert to filter pruning
    model = filter_prune(copy.deepcopy(model_baseline),torch.randn(1,3,224,224),output_transform=None,pruning_rates=best_candidates_prune)
    model = model.to(device)
    print(model)

    # calc params & flops
 
    sparse_params = calc_model_parameters(model)
    compression_sparse_params = sum(layer_params)/sum(sparse_params)

    sparse_flops = calc_model_flops(model, 224, mul_add=False)
    compression_sparse_flops = sum(layer_flops)/sum(sparse_flops)

    print('Model FLOPs = {:.2f} M | Sparse FLOPs = {:.2f} M | ( {:.2f}X | {:.2f}% )'.format(sum(layer_flops)/1000000, sum(sparse_flops)/1000000,compression_sparse_flops, (1.0-1.0/compression_sparse_flops)*100))
    print('Model Prams = {:.2f} M | Sparse Prams = {:.2f} M | ( {:.2f}X | {:.2f}% ) '.format(sum(layer_params)/1000000, sum(sparse_params)/1000000,compression_sparse_params, (1.0-1.0/compression_sparse_params)*100))


    if len(args.gpus) != 1:
        # model = nn.DataParallel(model, device_ids=args.gpus)
        model = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//8), model, dim=args.gpus[0], device_ids=args.gpus)
        cudnn.benchmark = True

    if args.resume == True:
        ckpt = torch.load(args.resume_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, ckpt))
        start_epoch =  ckpt['epoch']
        best_top1_acc = ckpt['best_top1_acc']
        best_top5_acc = ckpt['best_top5_acc']
        print("resume model epoch={:d} best_top1_acc={:.2f} best_top5_acc={:2f}".format(start_epoch,float(best_top1_acc),float(best_top5_acc)))
    else:
        if args.baseline == True:
            # # add baseline -> Few shot Learning (300 batchs) for quantization params
            model = init_params(model, trainLoader)
            model, avg_loss = Adaptive_BN(model, trainLoader)
            print(avg_loss)
        else:
            ## scrach ->  Adaptive-BN
            model,avg_loss = Adaptive_BN(model,trainLoader)

        init_top1,init_top5 = test(model, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        print('Model Init Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(init_top1),float(init_top5)))

    for pname, param in model.named_parameters():
        param.requires_grad = True

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad == True,model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad == True,model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

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