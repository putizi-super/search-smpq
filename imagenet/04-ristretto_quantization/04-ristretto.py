import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import time
from importlib import import_module

import sys
sys.path.append('..')
from data import imagenet_dali
from utils.options import parser
import utils.common as utils
from utils.balance import BalancedDataParallel
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models
# import torchvision.models as models
from modules.sq_module.sq_model import *
from modules.ristretto.ristretto_model import *

from tool.meter import *
from tool.mergebn import *
from tool.quant_ncnn import *

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

args = parser.parse_args()

# resnet50_caffe
w_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
a_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# pruning_rate=[0.2551811675419373, 0.10883450978339962, 0.22583542532487397, 0.19547508745002112, 0.11369726586315448, 0.16172124328818335, 0.157331220246815, 0.20782626185407252, 0.14391872824829183, 0.16731433895536404, 0.16290334019176939, 0.07652782086899101, 0.10724719780714069, 0.22040511663401713, 0.1833692878525247, 0.2356364290238544, 0.10622556191256613, 0.2113188295263494, 0.15299678063165925, 0.12861094312257557, 0.1634710847992596, 0.12435095588018703, 0.1421670802928368, 0.09432220564206424, 0.16910182541536306, 0.13505329597556698, 0.15073397260178548, 0.2088968274958775, 0.12160138166123047, 0.11254312615419304, 0.24582578448397757, 0.10120886520052749, 0.1716853042194909, 0.1179815795030305, 0.06870041233210827, 0.21524053356981737, 0.1920003803613265, 0.19109731990415102, 0.16537330873955833, 0.2374034855171861, 0.19063310269885037, 0.09511070407087727, 0.17917524639950763, 0.1522430506069873, 0.10205502360726787, 0.1797667607105435, 0.0974781111479703, 0.15958162315516056, 0.056207185648549735, 0.1495195241948409, 0.3634150746772293, 0.31189005300252604, 0.1266773460002937, 0.23611228824609895]
pruning_rate = [1.0]*len(w_bit)

# resnet18_caffe channel_keep /loss 3.1693
# w_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# a_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# pruning_rate=[0.2509892381838768, 0.1949955777182775, 0.15481105457277416, 0.181482879280324, 0.15236411504078137, 0.12624166044322543, 0.19945650833123205, 0.14867201615531073, 0.09744269325603885, 0.16507781611943093, 0.22461711527148995, 0.19174842125328845, 0.13530936319651926, 0.2769501456887007, 0.1281951215877744, 0.09779858333620649, 0.13675934743598922, 0.18702198771408263, 0.2592740954241761, 0.16431281448915383, 0.15246512653703692, 0.2972427183432819]
# pruning_rate = [1.0]*len(w_bit)

# mobilenet v2 channel_keep /loss 1.9658
# w_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# a_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# pruning_rate=[0.5358229688622911, 0.33571893075389764, 0.44285337575519834, 0.47113747224818237, 0.5960038730632095,0.6486709741045007, 0.36673321594974273, 0.8360254711441159, 0.43206469580395224, 0.5468655930901877, 0.6224618045092436, 0.5463259834144445, 0.29615942939349416, 0.5793754762611343, 0.6553775141708515, 0.4762178201910742, 0.7339470258354562, 0.39098359634468277, 0.5966725743598498, 0.6655638058362112, 0.5287433236204129, 0.41947665469918377, 0.56240719332682, 0.6026368647429946, 0.33118000145950666, 0.436275931018693, 0.6344299604445225, 0.27778293609974536, 0.4073726085165963, 0.44998150968290257, 0.7079136241818071, 0.325471243443323, 0.5007871133057589, 0.4834876821161839, 0.32421697228897334, 0.4440926207710644, 0.3999881768488689, 0.45919474704550184, 0.5319639333005075, 0.6443877419144484, 0.678641254675052, 0.5318928999075369, 0.4473333133694206, 0.5115358772717566, 0.444982499881899, 0.5256900665646667, 0.42366399006224853, 0.5321679802301845, 0.4755498437589937, 0.8226922845800333, 0.4531334228161739, 0.4703849299268835, 0.4800952337232943]
# pruning_rate = [1.0]*len(w_bit)

# mobilenet v1 channel_keep /loss 1.8011
# w_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# a_bit=[32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0]
# pruning_rate=[0.5446608306811687, 0.4706398979160246, 0.4306093718101938, 0.41478515048675746, 0.42849938347839267, 0.5395209122493769, 0.51392201048942, 0.6510646090729955, 0.41806565812678076, 0.3443596131443588, 0.586900700691442, 0.8508783214716091, 0.4665236913532016, 0.660833636760054, 0.3589439826462926, 0.5070963227912298, 0.37357237349717093, 0.41433391445566725, 0.5691640857341291, 0.5815423744006016, 0.5157306659435292, 0.6771761576950642, 0.4510926729345347, 0.5942457059849297, 0.5126241790094817, 0.6228088715686567, 0.4944682398377873, 0.522794214748819]

def get_model_featuremap(model, input):
    hook_list = []
    module_featuremap = []

    def featuremap_hook(self, input, output):
        module_featuremap.append(input[0])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(featuremap_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(featuremap_hook))

    model(input)

    for hook in hook_list:
        hook.remove()
    return module_featuremap

def saturation_weights(x,bit_width):
    max_data = (pow(2,bit_width-1)-1)
    min_data = -pow(2,bit_width-1)

    max_data = torch.Tensor([max_data]).to(device)
    min_data = torch.Tensor([min_data]).to(device)
    x= torch.max(min_data,torch.min(x,max_data)).to(device)
    return x

def saturation_activation(x,bit_width):
    max_data = (pow(2,bit_width)-1)
    min_data = 0

    max_data = torch.Tensor([max_data]).to(device)
    min_data = torch.Tensor([min_data]).to(device)
    x= torch.max(min_data,torch.min(x,max_data)).to(device)
    return x

def find_alpha_weights_per_layer(x,bw):
    max_data = torch.max(x)
    min_data = torch.min(x)
    if max_data > torch.abs(min_data):
        alpha = max_data
    else:
        alpha = torch.abs(min_data)
    return alpha

def find_alpha_weights_per_channel(x,bw):
    max_data = torch.max(x)
    min_data = torch.min(x)
    if max_data > torch.abs(min_data):
        alpha = max_data / pow(2,bw-1)
    else:
        alpha = torch.abs(min_data) / pow(2,bw-1)
    return alpha
    
def round_pass(x):
    y = x.round()
    y_grad = x
    # y_grad = torch.clamp(x, -1.0, 1.0)
    return (y - y_grad).detach() + y_grad


def weight_init(model_baseline, model, trainLoader, epoch):
    model_baseline.eval()
    model.eval()
    cnt = 0
    alpha_weight = []
    weights = []
    for m in model_baseline.modules():
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            weight = copy.deepcopy(m.weight.data)
            weights.append(weight)
            alpha_weight.append(find_alpha_weights_per_layer(weight,args.avg_bw_weights))
            cnt += 1

    alpha_weight = Parameter(torch.Tensor(alpha_weight).to(device))

    loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    # optimizer_mes_weights = optim.Adam([alpha_weight], lr=0.01, weight_decay=4e-5)
    optimizer_mes_weights = torch.optim.SGD([alpha_weight], lr=0.1, momentum=0.9, weight_decay=4e-5)  
    scheduler_weights = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mes_weights, min_lr=1e-5, verbose=False, patience=200)
    for i in range(epoch):
        # Adeptive Weights Step Size
        for batch, batch_data in enumerate(trainLoader):
            if batch < 600:
                optimizer_mes_weights.zero_grad()
                weight_loss = 0
                
                for i in range(cnt):
                    weight = weights[i]
                    weight_f = weight.to(device)
                    scale_w = (2.0 ** round_pass(torch.log2(alpha_weight[i])) / (2.0 ** (args.avg_bw_weights-1)))
                    weight = weight / scale_w
                    weight = torch.clamp(weight, - 2 ** (args.avg_bw_weights - 1), 2 ** (args.avg_bw_weights - 1) -1 )
                    weight = round_pass(weight)
                    weight_q = (weight * scale_w).to(device)
                    weight_loss += loss_mse(weight_f,weight_q)
 
                print('Weight Loss = ' + str(weight_loss.data)) 
                weight_loss.backward()
                optimizer_mes_weights.step()
                scheduler_weights.step(weight_loss.item())
           
            else:
                break
        
        cnt = 0
        for m in model.modules():
            if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
                m.w_alpha.data = alpha_weight[cnt].data.to(device)
                cnt += 1

    print((torch.round(torch.log2(alpha_weight))-7))
    return alpha_weight
    
def activation_init(model_baseline, model, trainLoader, epoch):
    model_baseline.eval()
    model.eval()
    cnt = 0
    for m in model_baseline.modules():
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            cnt += 1

    alpha_activation = Parameter(3*torch.ones(cnt).to(device))

    loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    # optimizer_mes_act = optim.Adam([alpha_activation], lr=0.01, weight_decay=4e-5)  
    optimizer_mes_act = torch.optim.SGD([alpha_activation], lr=0.1, momentum=0.9, weight_decay=4e-5)  
    scheduler_act = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mes_act, min_lr=1e-5, verbose=False, patience=200)
    for i in range(epoch):
        for batch, batch_data in enumerate(trainLoader):
            if batch < 1000:
                inputs = batch_data[0]['data'].to(device)
                targets = batch_data[0]['label'].squeeze().long().to(device)
                data = get_model_featuremap(model_baseline,inputs)
                data_sq = get_model_featuremap(model,inputs)
                optimizer_mes_act.zero_grad()
                cnt = 0
                fm_loss = 0
                
                for x in data:
                    x_f = x.to(device)
                    x_q = data_sq[cnt].data.to(device)
                    scale = (2.0 ** round_pass(torch.log2(alpha_activation[cnt])) / (2.0 ** (args.avg_bw_fm-1)))
                    x_q = x_q / scale
                    x_q = torch.clamp(x_q, - 2 ** (args.avg_bw_fm - 1), 2 ** (args.avg_bw_fm - 1) -1 )
                    x_q = round_pass(x_q)
                    x_q = x_q * scale
                    fm_loss += loss_mse(x_f,x_q)
                    cnt = cnt + 1
                print('Activation Loss = '+str(fm_loss.data))
                fm_loss.backward()
                optimizer_mes_act.step()
                scheduler_act.step(fm_loss.item())
           
            else:
                break
    print((torch.round(torch.log2(alpha_activation))-7))
    return alpha_activation



def init_params(model, trainLoader):
    model.train()

    for m in model.modules():
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            for pname, param in m.named_parameters():
                if ("w_alpha" in pname) or ("a_alpha" in pname):
                    param.requires_grad = True
                else:
                    param.requires_grad = True
        else:
            for pname, param in m.named_parameters():
                param.requires_grad = False

    # print(filter(lambda p: p.requires_grad == True, model.parameters()))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=1e-5, weight_decay=1e-5)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, verbose=False, patience=200)
    # criterion_kd = utils.DistributionLoss()
    # loss_mse = torch.nn.MSELoss(reduce=True, size_average=True)

    for batch_idx, batch_data in enumerate(trainLoader):
        if batch_idx <= 500:
            inputs = batch_data[0]['data'].to(device)
            targets = batch_data[0]['label'].squeeze().long().to(device)
            optimizer.zero_grad()
            output = model(inputs)
            total_loss = loss_func(output, targets)
            print(total_loss)
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
        else:
            break
    return model

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

def main():

    print('==> Building model..')
    model = models.__dict__[args.arch]().to(device)
    # model = models.__dict__[args.arch](pretrained=True).to(device)
 
    if args.baseline == True:
        model_baseline= models.__dict__[args.arch]().to(device)
        # model_baseline = convert_to_sqconv(model_baseline, w_bit, a_bit, pruning_rate).to(device)
        ckpt = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(utils.convert_keys(model_baseline, ckpt))
        # best_top1_acc = ckpt['best_top1_acc']
        # best_top5_acc = ckpt['best_top5_acc']
        # print("model best_top1_acc={:.2f} best_top5_acc={:2f}".format(float(best_top1_acc),float(best_top5_acc)))

    best_top1_acc, best_top5_acc = test(model_baseline, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    
    
    # model_state_dict = model_baseline.module.state_dict() if len(args.gpus) > 1 else model_baseline.state_dict()
    # state = {
    #     'state_dict': model_state_dict,
    #     'best_top1_acc': best_top1_acc,
    #     'best_top5_acc': best_top5_acc
    # }
    # checkpoint.save_model(state, True)
    # sys.exit()
    
    params = utils.get_params_model(model_baseline)
    model = convert_to_ristretto(copy.deepcopy(model_baseline), [8]*len(w_bit), [32]*len(a_bit), pruning_rate).to(device)
    model = utils.load_params_model(model,params)

    alpha_weight = weight_init(model_baseline, model, trainLoader, 1)
    # model = utils.load_params_model(model,params)
    
    cnt = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            m.w_alpha.data = alpha_weight[cnt].data.to(device)
            cnt += 1

    model, _ = Adaptive_BN(model, trainLoader)
    params = utils.get_params_model(model)
    alpha_activation = activation_init(model_baseline, model, trainLoader, 1)
    
    # 11.5661 torvision
    # alpha_weight = torch.Tensor([ 0.3634, 32,  1.4136,  0.7055,  5.6616,  0.7407,  0.3566,  5.6650,  1.4138,  0.3590,  5.6605,  0.7193,  0.1807,  5.6777,  0.7350,  0.1769, 5.6603,  0.7441,  0.1864,  2.8290,  0.7086,  0.1083,  5.6582,  0.7028,  0.0884, 11.3140,  0.7071,  0.0885,  5.6571,  0.7077,  0.1083,  5.6611, 0.3620,  0.1382, 11.3049,  0.3660,  0.1735,  5.7479,  0.7156,  0.1441, 2.8279,  0.1909,  0.1806, 11.3080,  0.3119,  0.1618,  5.6627,  0.3597, 0.0888,  7.7831,  0.6961,  1.4173,  0.2004])
    # alpha_activation = torch.Tensor([2.8657, 2.8284, 5.6569, 5.6298, 2.8285, 2.8285, 2.8285, 2.2366, 2.3661, 5.6567, 2.2539, 2.3303, 2.8285, 2.2329, 2.2474, 2.8285, 2.2353, 2.2494, 2.8453, 2.2408, 2.2712, 2.3542, 2.2333, 2.2467, 2.4887, 2.2377, 2.2459, 2.8284, 2.2417, 2.3632, 2.8284, 2.2391, 2.2687, 2.2913, 2.2435, 2.3044, 2.8283, 2.2681, 2.3347, 5.0877, 2.3514, 2.8290, 5.0666, 2.4488, 2.8284, 5.6570, 2.8113, 2.8284, 7.2245, 2.3159, 2.2325, 2.2266, 2.8352])

    # alpha_weight = torch.Tensor([ 0.7070, 11.3162,  1.4140,  0.7126,  5.6570,  0.8578,  0.7075,  5.9426, 1.4288,  0.3568,  3.2038,  0.7778,  0.1893, 11.3155,  0.7755,  0.1771, 5.7202,  1.4114,  0.2370,  2.9229,  0.7083,  0.1786,  5.6561,  1.4143, 0.1779,  5.6563,  0.7086,  0.1748,  5.6604,  0.7068,  0.1753,  5.6576, 0.4652,  0.1761,  5.6610,  0.7969,  0.1623,  5.6574,  0.7091,  0.1775, 2.8299,  0.3544,  0.1570, 11.3147,  0.3541,  0.1554,  5.6568,  0.5024, 0.1725,  2.8307,  0.3626,  1.4142,  0.5175])
    # alpha_activation = torch.Tensor([ 2.9404,  5.6557,  5.6876,  5.6572,  5.6539,  2.8725,  5.6556,  1.4146, 2.8293,  5.6652,  2.8275,  2.8285,  2.9635,  1.4154,  1.7165,  5.6580, 1.4149,  1.4461,  5.6614,  1.6618,  2.8305,  2.8367,  1.4130,  1.4146, 2.8564,  0.7429,  1.4189,  2.9851,  1.4122,  1.4413,  3.3347,  1.4161, 2.8297,  2.8292,  1.3422,  2.8278,  2.8467,  1.4122,  2.8296,  5.6564, 1.4229,  5.6562,  5.6513,  1.5508,  2.8293,  5.6569,  1.4210,  2.8356,  11.3136,  1.4145,  1.4198,  1.4984,  5.6578])

    ristretto_cnt[0] = 0
    model = convert_to_ristretto(copy.deepcopy(model_baseline),[8]*len(w_bit), [8]*len(a_bit), pruning_rate).to(device)
    model = utils.load_params_model(model,params)


    cnt = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            m.w_alpha.data = alpha_weight[cnt].data.to(device)
            m.a_alpha.data = alpha_activation[cnt].data.to(device)
            cnt += 1
    

    # model = init_params(model, trainLoader)
    model, _ = Adaptive_BN(model, trainLoader)

    if len(args.gpus) != 1:
        model_baseline = BalancedDataParallel(int(args.train_batch_size//len(args.gpus)//2), model_baseline, dim=args.gpus[0], device_ids=args.gpus)
        cudnn.benchmark = True
    ristretto_top1_acc, ristretto_top5_acc = test(model, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    # model_state_dict = model.state_dict()

    state = {
        'state_dict': model_state_dict,
        'ristretto_top1_acc': ristretto_top1_acc,
        'ristretto_top5_acc': ristretto_top5_acc
    }
    checkpoint.save_model(state, True)
    
    logger.info('Quantization Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3}'.format(float(ristretto_top1_acc), float(ristretto_top5_acc)))


    del(model_baseline)

if __name__ == '__main__':
    main()