import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.nn.modules import loss

import os
import math
import time
from importlib import import_module

import sys
sys.path.append('../../')
from data import imagenet_dali
from utils.options import args
from utils.balance import BalancedDataParallel
import utils.common as utils
from loss.kd_loss import *
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

from modules.sq_module.sq_model import *
from modules.sq_module.filter_pruning import *

from tool.meter import *

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss
if args.mixup > 0.0:
    loss_func = lambda: NLLMultiLabelSmooth(args.label_smoothing)
elif args.label_smoothing > 0.0:
    loss_func = lambda: LabelSmoothing(args.label_smoothing)
loss_func = loss_func()
criterion_kd = DistributionLoss()

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

# # # # ### ResNet18 w=3 a=3 # 
# w_bit=[8.0, 7.0, 6.0, 7.0, 8.0, 5.0, 3.0, 4.0, 3.0, 6.0, 3.0, 4.0, 3.0, 4.0, 6.0, 3.0, 2.0, 3.0, 3.0, 1.0, 8.0]
# a_bit=[6.0, 3.0, 3.0, 2.0, 2.0, 2.0, 3.0, 2.0, 5.0, 1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 4.0, 2.0, 2.0, 5.0, 7.0, 8.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # # # # ### ResNet18 w=2 a=2 epoch=60 Top1=66.88% Top5=87.07% 
# w_bit=[7.0, 6.0, 4.0, 2.0, 5.0, 6.0, 5.0, 5.0, 4.0, 2.0, 4.0, 4.0, 4.0, 2.0, 2.0, 3.0, 1.0, 5.0, 1.0, 1.0, 6.0]
# a_bit=[6.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 5.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # # # # ### ResNet18 w=2 a=2 epoch=60 Top1=67.39% Top5=87.66% 
# w_bit=[6.0, 4.0, 5.0, 6.0, 5.0, 5.0, 3.0, 4.0, 6.0, 5.0, 3.0, 5.0, 4.0, 3.0, 3.0, 2.0, 1.0, 4.0, 1.0, 1.0, 5.0]
# a_bit=[4.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 4.0, 3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 6.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# # # # ### ResNet18 w=2 a=3 epoch=60 Top1=  Top5= 
# w_bit=[5.0, 5.0, 4.0, 5.0, 3.0, 4.0, 3.0, 5.0, 6.0, 5.0, 3.0, 3.0, 6.0, 4.0, 3.0, 2.0, 1.0, 5.0, 1.0, 1.0, 6.0]
# a_bit=[5.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 5.0, 4.0, 6.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# w_bit = [32.0]+[2.0]*19+[32.0]
# a_bit = [32.0]+[2.0]*19+[32.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # # # # ### ResNet50 F=6 P=6
# w_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# a_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# pruning_rate=[0.4544799499238741, 0.17293111118470417, 0.10570434574049917, 0.09671292096581814, 0.34075498096008033, 0.21218838811449178, 0.13339259543592502, 0.08902489301743159, 0.034742738476600664, 0.21114029211850252, 0.01610456318062021, 0.0963024864494802, 0.05960583830142748, 0.33237912734626757, 0.06557601652064513, 0.2344851343329461, 0.3476908337088134, 0.025319112619392, 0.03471023838858935, 0.2847967105016298, 0.13962922385567095, 0.383470933914414, 0.08607429124991955, 0.14904735598976382, 0.09224329874879264, 0.0235245239444216, 0.03171706154763519, 0.03311132942986696, 0.45400978021981725, 0.2483381839695284, 0.3023673631867798, 0.09665479650375274, 0.056830200401948874, 0.24605219591787567, 0.09866243218046379, 0.08442867105294122, 0.21069220694996443, 0.2420510539099943, 0.030837752106546747, 0.07836703363511033, 0.3403835387909439, 0.04263480289452648, 0.08463995979828495, 0.14871560414300897, 0.29966469888727976, 0.05238297471816625, 0.12181775556583854, 0.18734005707909177, 0.12557022822686764, 0.1086836058489306, 0.24126523909613626, 0.21195730964369436, 0.16572683733474586, 0.2093529927318662]

# # # # # ### ResNet50 w=2 a=2  epoch=60 Top1=71.953% Top5=90.409% (search loss=3.48 top1=39.75%,top5=66.65%)
# w_bit=[7.0, 2.0, 4.0, 6.0, 2.0, 2.0, 3.0, 4.0, 1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 3.0, 1.0, 3.0, 3.0, 2.0, 3.0, 1.0, 2.0, 1.0, 6.0, 6.0, 4.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 5.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 2.0, 1.0, 4.0]
# a_bit=[5.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 6.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# # # # # ### ResNet50 w=2 a=4  epoch=60 Top1=71.953% Top5=90.409% (search loss=3.48 top1=39.75%,top5=66.65%)
w_bit=[5.0, 6.0, 2.0, 3.0, 4.0, 3.0, 2.0, 4.0, 5.0, 6.0, 2.0, 4.0, 3.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 5.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 4.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 2.0, 3.0]
a_bit=[6.0, 3.0, 5.0, 4.0, 4.0, 4.0, 5.0, 5.0, 3.0, 6.0, 4.0, 4.0, 2.0, 3.0, 4.0, 3.0, 5.0, 2.0, 6.0, 4.0, 2.0, 4.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 5.0, 5.0, 4.0, 5.0, 4.0, 3.0, 4.0, 2.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 5.0, 3.0, 4.0, 5.0, 4.0, 2.0, 3.0, 3.0, 5.0, 4.0, 5.0, 5.0]
pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# # # # # ### MobileNet_v1 w=4 a=4 epoch=60 Top1=69.67% Top5=88.74% 
# w_bit=[6.0, 6.0, 5.0, 4.0, 5.0, 5.0, 4.0, 6.0, 5.0, 5.0, 5.0, 4.0, 4.0, 6.0, 3.0, 4.0, 3.0, 5.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 4.0, 4.0, 3.0, 5.0]
# a_bit=[6.0, 5.0, 5.0, 3.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 5.0, 3.0, 5.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 3.0, 5.0, 4.0, 4.0, 6.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# # # # # ### MobileNet_v2 w=4 a=4 epoch=60 Top1=70.632 % Top5= 89.336% 
# w_bit=[8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0, 5.0, 3.0, 5.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 4.0, 3.0, 4.0, 3.0, 5.0, 5.0, 4.0, 5.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 5.0, 3.0, 4.0, 5.0, 5.0, 3.0, 4.0]
# a_bit=[8.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.0, 5.0, 4.0, 3.0, 4.0, 4.0, 8.0]
# pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def train(model_teacher, model, optimizer, trainLoader, args, epoch, topk=(1,)):
    model_teacher.eval()
    model.train()
    losses = utils.AverageMeter()
    losses_ce = utils.AverageMeter()
    losses_kd = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = trainLoader._size // args.train_batch_size // 10
    start_time = time.time()
    # ealy stop 
    if epoch >= 20:
        alpha1 = 1.0
    else:
        alpha1 = 0.0
    # alpha1 = utils.step(epoch,args.num_epochs)
    # alpha1 = utils.const(num=0.5)
    for batch, batch_data in enumerate(trainLoader):
        inputs = batch_data[0]['data'].to(device)
        targets = batch_data[0]['label'].squeeze().long().to(device)

        lr = adjust_learning_rate(optimizer, epoch, batch, trainLoader._size // args.train_batch_size)

        optimizer.zero_grad()
        output = model(inputs)
        output_teacher = model_teacher(inputs)
        loss_ce = loss_func(output, targets)
        loss_kd = criterion_kd(output, output_teacher)
        loss = 0.5*loss_ce+0.5*loss_kd
        # loss = alpha1*loss_ce+(1.0-alpha1)*loss_kd
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        losses_ce.update(loss_ce.item(), inputs.size(0))
        losses_kd.update(loss_kd.item(), inputs.size(0))
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
                'Loss_CE {:.4f}\t'
                'Loss_KD {:.4f}\t'
                'lr {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, trainLoader._size,
                    float(losses.avg),float(losses_ce.avg),float(losses_kd.avg),lr, float(accuracy.avg), float(top5_accuracy.avg), cost_time
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

    for batch_idx, batch_data in enumerate(trainLoader):
        inputs = batch_data[0]['data'].to(device)
        targets = batch_data[0]['label'].squeeze().long().to(device)
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
        for batch_idx, batch_data in enumerate(trainLoader):
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            if batch_idx <= 100:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_func(output, targets)
                losses.update(loss.item(), inputs.size(0))
            else:
                break
    return model, losses.avg


def main():
    start_epoch = 0
    best_top1_acc = 0.0
    best_top5_acc = 0.0

    print('==> Building model..')
    model_baseline = models.__dict__[args.arch]().to(device)
    model_teacher = models.__dict__[args.arch_teacher]().to(device)
 
    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location=device)
        ckpt_teacher = torch.load(args.teacher_model, map_location=device)
        model_baseline.load_state_dict(utils.convert_keys(model_baseline, ckpt))
        model_teacher.load_state_dict(utils.convert_keys(model_teacher, ckpt_teacher))
        # model_baseline_top1,model_baseline_top5 = test(model_baseline, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        # print('Model Baseline Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(model_baseline_top1),float(model_baseline_top5)))
        params = utils.get_params_model(model_baseline)

    for p in model_baseline.parameters():
        p.requires_grad = False
    model_baseline.eval()

    # convert to sqconv
    model = convert_to_sqconv(model_baseline,w_bit, a_bit, pruning_rate).to(device)

    print(model)

    # calc featuremap & params & flops
    layer_featuremap = calc_model_featuremap(model,224)
    quatization_featuremap =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_featuremap,a_bit))))
    compression_featuremap = sum(layer_featuremap)*32/quatization_featuremap

    layer_params = calc_model_parameters(model)
    quatization_params =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_params,w_bit))))
    compression_quantization_params = sum(layer_params)*32/quatization_params

    sparse_params =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_params,pruning_rate))))
    compression_sparse_params = sum(layer_params)/sparse_params

    layer_flops = calc_model_flops(model, 224, mul_add=False)
    sparse_flops =sum(list(map(lambda x :x[0]*x[1] ,zip(layer_flops,pruning_rate))))
    compression_sparse_flops = sum(layer_flops)/sparse_flops

    print('Model FLOPs      = {:.2f} M         | Sparse       FLOPs      = {:.2f} M         | {:.2f} X '.format(sum(layer_flops)/1000000, sparse_flops/1000000,compression_sparse_flops))
    print('Model Prams      = {:.2f} M (num)   | Sparse       Prams      = {:.2f} M  (num)  | {:.2f} X '.format(sum(layer_params)/1000000, sparse_params/1000000,compression_sparse_params))
    print('Model Prams      = {:.2f} M (Byte)  | Quantization Prams      = {:.2f} M  (Byte) | AvgBit = {:.2f} bit | {:.2f} X | '.format(sum(layer_params*4)/1000000, quatization_params/8/1000000,32/compression_quantization_params,compression_quantization_params))
    print('Model FeatureMap = {:.2f} M (Byte)  | Quantization FeatureMap = {:.2f} M  (Byte) | AvgBit = {:.2f} bit | {:.2f} X | '.format(sum(layer_featuremap*4)/1000000, quatization_featuremap/8/1000000,32/compression_featuremap,compression_featuremap))

    if len(args.gpus) != 1:
        # model = nn.DataParallel(model, device_ids=args.gpus)
        model_baseline = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//1.5), model_baseline, dim=args.gpus[0], device_ids=args.gpus)
        model = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//1.5), model, dim=args.gpus[0], device_ids=args.gpus)
        model_teacher = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//1.5), model_teacher, dim=args.gpus[0], device_ids=args.gpus)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    if args.resume == True:
        ckpt = torch.load(args.resume_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, ckpt))
        start_epoch =  ckpt['epoch']
        best_top1_acc = ckpt['best_top1_acc']
        best_top5_acc = ckpt['best_top5_acc']
        print("resume model epoch={:d} best_top1_acc={:.2f} best_top5_acc={:2f}".format(start_epoch,float(best_top1_acc),float(best_top5_acc)))

    else:
        if args.baseline == True:
            model = utils.load_params_model(model,params)
            # # add baseline -> Few shot Learning (1000 batchs) for quantization params
            model = init_params(model_baseline, model, w_bit, trainLoader)
            model, avg_loss = Adaptive_BN(model, trainLoader)
            print(avg_loss)
        else:
            ## scrach ->  Adaptive-BN
            model,avg_loss = Adaptive_BN(model,trainLoader)

        init_top1,init_top5 = test(model, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        print('Model Init Top1 = {:.2f}% | Top5 = {:.2f}% '.format(float(init_top1),float(init_top5)))

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
    # print(k1)        
    # print(b1)
    # print(k2)      
    # print(b2)  


    for pname, param in model.named_parameters():
        param.requires_grad = True

    # all_parameters = model.parameters()
    # weight_parameters = []
    # for pname, p in model.named_parameters():
    #     if p.ndimension() == 4 or 'conv' in pname:
    #         weight_parameters.append(p)
    # weight_parameters_id = list(map(id, weight_parameters))
    # other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    # if args.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(
    #             [{'params' : other_parameters},
    #             {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #             lr=args.lr,momentum=args.momentum)
    # elif args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #             [{'params' : other_parameters},
    #             {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
    #             lr=args.lr)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad == True,model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad == True,model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.num_epochs):

        train(model_teacher,model, optimizer, trainLoader, args, epoch, topk=(1, 5))

        test_top1_acc, test_top5_acc = test(model, testLoader, topk=(1, 5))

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
        # print(k1)        
        # print(b1)
        # print(k2)      
        # print(b2) 

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