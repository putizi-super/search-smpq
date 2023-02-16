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
import tool.pruning as tp

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

# # # googelenet 
# best_candidates_prune=[0.5783892918570721, 0.44214892201171513, 0.41783040829940993, 0.4243916376398989, 0.25600050600056246, 0.7731664358949468, 0.424093100823699, 0.3914272155655246, 0.5844972360650882, 0.211317374631301, 0.16393342750907683, 0.5115930511062365, 0.6950926666676523, 0.3112272097612428, 0.3167998211078999, 0.39263853648826763, 0.5197150486509933, 0.41756578285920254, 0.4263944004556821, 0.9061254840618091, 0.2791298090364277, 0.440746937784797, 0.5672043281462147, 0.6506410433144982, 0.3799923453968092, 0.6218495953545248, 0.8888327664584921, 0.37192898218957626, 0.36924979247625495, 0.4470575646407105, 0.46409914739895197, 0.3899841062358186, 0.5054974751139547, 0.9999, 0.27027295874481694, 0.4057959298093796, 0.42309212615670655, 0.3289948636031054, 0.4657675852222975, 0.4404246144531849, 0.21399312711914015, 0.5957727219567037, 0.6789374122651937, 0.46606923622502794, 0.23517717032097996, 0.22105980400903064, 0.5934526006474053, 0.5574012411173462, 0.2235297904003961, 0.20476101279670395, 0.36871533281828917, 0.41412499199780556, 0.42373792492648843, 0.7778231863604189, 0.3394453826278556, 0.6346002274168755, 0.3316069967988262, 0.4200521587376907, 0.647769649914988, 0.5185988507038538, 0.441304467295753, 0.23206739516489952, 0.6214789087986057, 0.14290897942042508]

# # # resnet20_B #52%
# best_candidates_prune=[0.10631335286980659, 0.2602891071738058, 0.05188991905797194, 0.28081476842975706, 0.16207180444416142, 0.28540366406797085, 0.12950179753776633, 0.18969047638856115, 0.17051554493668647, 0.12905499922512031, 0.5135201920578154, 0.11400679644050868, 0.251547434883359, 0.10938851766082949, 0.07249558702079231, 0.0883920651488485, 0.21766080637569404, 0.16943616750065196, 0.25183465025754, 0.2865192166124342, 0.1360891533961404]
# best_candidates_prune=[0.1057541529324978, 0.2752305783312998, 0.02171106593874087, 0.32489539669343254, 0.2122742457879765, 0.20988830578115447, 0.15673659470317566, 0.27054305487972596, 0.1700199040852207, 0.08218849050293198, 0.44205378027275677, 0.01399567153961695, 0.3137980654926439, 0.12143020159595097, 0.05791889367803603, 0.0766978795586413, 0.15975535166625776, 0.13391799326970338, 0.25153067499467563, 0.3228186015348077, 0.13216923127519165]

# # # resnet56_B #52%
# best_candidates_prune=[0.065868243665229, 0.06612021359573102, 0.06514570954456793, 0.2446322605616501, 0.00440388461700194, 0.15248610777443883, 0.06060342017844688, 0.03513188273126743, 0.0452085069345741, 0.12485249201599444, 0.10258611437712459, 0.16134253235706048, 0.0042721543234252694, 0.10694604572627839, 0.2646027637845273, 0.18105356709518217, 0.07026107569106804, 0.27215662210012515, 0.11827363615419557, 0.09635548506587026, 0.32986401427997103, 0.07405347867258481, 0.19550870091548073, 0.0181195488852169, 0.06506611083373345, 0.015704114873491628, 0.029963892905639583, 0.09130769479565531, 0.09271223464503456, 0.015964689892838814, 0.1559756828468809, 0.042340347656817205, 0.05079775962982443, 0.19070594651881306, 0.06213633173336838, 0.1290917993567715, 0.1305280084748507, 0.028343419809152975, 0.06619027927152474, 0.11875390292251205, 0.109877269492099, 0.18821526324253368, 0.18548067490024145, 0.13793609558896255, 0.011966236491932594, 0.17622595790487272, 0.11452032509719445, 0.037378047848047095, 0.06653569197725599, 0.18798253643982057, 0.026244095384116827, 0.022157297932367225, 0.10374339028350002, 0.1706250042811147, 0.007517048203215457, 0.12994577823759276, 0.0596871716277855]
best_candidates_prune=[0.14537810651170582, 0.07046535821173114, 0.03138930160845296, 0.20401619360178636, 0.04898748369870838, 0.03548502109114553, 0.03499636847410009, 0.24754967819561124, 0.04655459120077239, 0.2750116643575661, 0.47842791706350773, 0.10088709638200155, 0.08810135328618825, 0.17998410231315629, 0.026070575691798624, 0.122673128097366, 0.09677609377621146, 0.037163671089681764, 0.06540131440864451, 0.0474654916324367, 0.05398064546642867, 0.29715650046357894, 0.1564158610293075, 0.0001, 0.31313925078271854, 0.0893512057155406, 0.18039643208116468, 0.13626452131568603, 0.054165452869037456, 0.035205333045130995, 0.3013679529245804, 0.10163650112528887, 0.040501617172739146, 0.03811807369414302, 0.3511827481709898, 0.077693869738138, 0.21165928111563004, 0.026327692600191882, 0.14711889970926362, 0.02601971187973233, 0.0076247363289160975, 0.02627681474915059, 0.07686357082335621, 0.08785225985907698, 0.0001, 0.07536167614382452, 0.12541019573382417, 0.00892249009456681, 0.13304660508736244, 0.2070860323019339, 0.08117819045681567, 0.09249197390709842, 0.0878648607984997, 0.1521240563090506, 0.05903936740625862, 0.08785919168381012, 0.019148868951361146]

# # # vgg16_bn #50%
# best_candidates_prune=[0.3818597463364703, 0.2797618000570214, 0.18609367206419922, 0.14824171347520804, 0.21153995369429995, 0.1736370390216238, 0.1450611182319534, 0.35447730842230585, 0.6278287544028218, 0.7378522633852898, 0.33083419110957846, 0.28160308050369653, 0.23424168105078091]
# # #  vgg16_bn #70%
# best_candidates_prune=[0.4297042385741193, 0.31826920999795943, 0.5489678871609975, 0.10336288657519789, 0.41640116950504985, 0.21997449558862692, 0.5813422043291151, 0.6005905894439321, 0.7812183716028794, 0.6905465444253638, 0.435398263926202, 0.4802643262531826, 0.35961762219010307]

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
 
    model = filter_prune(copy.deepcopy(model_baseline),torch.randn(1,3,32,32),output_transform=None,pruning_rates=best_candidates_prune)
    model = model.to(device)
    print(model)

    # calc params & flops
 
    sparse_params = calc_model_parameters(model)
    compression_sparse_params = sum(layer_params)/sum(sparse_params)

    sparse_flops = calc_model_flops(model, 32, mul_add=False)
    compression_sparse_flops = sum(layer_flops)/sum(sparse_flops)

    print('Model FLOPs = {:.2f} M | Sparse FLOPs = {:.2f} M | ( {:.2f}X | {:.2f}% )'.format(sum(layer_flops)/1000000, sum(sparse_flops)/1000000,compression_sparse_flops, (1.0-1.0/compression_sparse_flops)*100))
    print('Model Prams = {:.2f} M | Sparse Prams = {:.2f} M | ( {:.2f}X | {:.2f}% ) '.format(sum(layer_params)/1000000, sum(sparse_params)/1000000,compression_sparse_params, (1.0-1.0/compression_sparse_params)*100))

    if args.baseline == True:
        # model = load_params_model_fp(model,params)
        # model = load_params_pruned_resnet_l2(model, state_dict_baseline)
        # model = load_params_pruned_resnet(model, state_dict_baseline)
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