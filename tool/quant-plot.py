import torch
import torch.nn as nn
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import time
from importlib import import_module
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('..')

from data import imagenet_dali
import utils.common as utils
from utils.options import args
from utils.balance import BalancedDataParallel
import models.imagenet as models
from modules.sq_module.sq_model import *

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'


# # ### ResNet18 w=2 a=2 epoch=60 Top1=66.88% Top5=87.07% 
w_bit=[7.0, 6.0, 4.0, 2.0, 5.0, 6.0, 5.0, 5.0, 4.0, 2.0, 4.0, 4.0, 4.0, 2.0, 2.0, 3.0, 1.0, 5.0, 1.0, 1.0, 6.0]
a_bit=[6.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 5.0]
pruning_rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

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

def pruning(x,rate):
    k_num = int((1.0-rate) * len(x.view(-1)))
    if k_num >= len(x.view(-1)):
        k_num = len(x.view(-1))-1
    elif k_num < 0:
        k_num = 0
    input_s = torch.sort(x.view(-1))
    val = input_s[0][k_num] 
    mask = x > val
    return mask.float()

def quan_activation(x, a_s, a_beta, a_bit, unilateral=True):
    if a_bit == 32:
        activation = x
    elif a_bit == 1:
        activation = torch.sign(x)
    else:
        x = (x - a_beta ) / a_s
        if unilateral == True:
            x = torch.clamp(x, 0, 2**(a_bit)-1)
        else:
            x = torch.clamp(x, -2**(a_bit-1) , 2**(a_bit-1)-1)
        x = torch.round(x)
        activation = x * a_s + a_beta
    return activation

def quan_weight(x, w_s, w_bit, pruning_rate):
    if  w_bit == 32:
        if pruning_rate == 1.0:
            weight_q = x
        else:
            mask = pruning(torch.abs(x),pruning_rate)
            x = x * mask
            weight_q = x
    elif w_bit == 1:
        if pruning_rate == 1.0:
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(x),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        else:
            mask = pruning(torch.abs(x),pruning_rate)
            x = x * mask
            weights_sum = torch.sum(torch.sum(torch.sum(abs(x),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            mask_sum = torch.sum(torch.sum(torch.sum(mask,dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            scaling_factor = (weights_sum / (mask_sum+1e-12))
        weight_q = scaling_factor * torch.sign(x)
    else:
        if  pruning_rate != 1.0:
            mask = pruning(torch.abs(x), pruning_rate)
            x = x * mask
        x = x / w_s
        x = torch.clamp(x, -2**(w_bit-1) , 2**(w_bit-1)-1)
        x = x.round()
        weight_q = x * w_s
    return weight_q

def get_model_weight(model):
    params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            params.append(m.weight.data)
    return params

def get_model_featuremap(model, input):
    hook_list = []
    module_featuremap = []

    def conv_hook(self, input, output):
        module_featuremap.append(input[0])

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # if isinstance(m, Scale_HardTanh):
            hook_list.append(m.register_forward_hook(conv_hook))

    model(input)

    for hook in hook_list:
        hook.remove()
    return module_featuremap

def main():
    model_baseline= models.__dict__[args.arch]().to(device)
    print(model_baseline)


    if args.baseline == True:
        model = convert_to_sqconv(model_baseline,w_bit, a_bit, pruning_rate).to(device)
        checkpoint = torch.load(args.baseline_model, map_location=device)
        model.load_state_dict(utils.convert_keys(model, checkpoint))

    w_s=[]
    a_s=[]
    a_beta=[]
    for name, m in model.named_modules():
        if (isinstance(m, SQConv)) or (isinstance(m, SQLinear)):
            w_s.append(m.w_s.data)
            a_s.append(m.a_s.data)
            a_beta.append(m.a_beta.data)

    name=[]
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            name.append(n)

    # if ("conv.0" in pname) or ("conv.1" in pname) or ("conv1.0" in pname):
    come = []
    for a in name:
        str1 = a[:a.find('.weight')]
        come.append(str1)
    # print(come)


    param = get_model_weight(model)

    
    # fig,axes = plt.subplots(ncols=7,nrows=3)
    # ax = axes.ravel()
    
    # # weights histogram
    # for index in range(len(param)):
    #     # print('weight',index)
    #     weights = param[index]
    #     quant_w = quan_weight(param[index], w_s[index],w_bit[index],pruning_rate[index])
    #     params = weights.view(-1).detach().cpu().numpy()
    #     quant_params = quant_w.view(-1).detach().cpu().numpy()
    #     ax[index].set_title(name[index])
    #     ax[index].hist(params,bins=100,color='steelblue',density=False)
    #     qaun_ax  = ax[index].twinx()
    #     qaun_ax.hist(quant_params,bins=100,color='red',density=False)
    #     ax[index].set_xticks([])
    #     ax[index].set_yticks([])
    #     qaun_ax.set_xticks([])
    #     qaun_ax.set_yticks([])

    # fig.tight_layout(pad=0.2, w_pad=0.0, h_pad=0.0)
    # # plt.tight_layout()
    # plt.show()
    # fig.savefig('resnet18_1.pdf',dpi=600,format='pdf')
    # print('weight hist saved!')

    # sys.exit()

    #fig1,axes1 = plt.subplots(ncols=4,nrows=5)
    fig1,axes1 = plt.subplots(ncols=7,nrows=3)
    ax1 = axes1.ravel()
    
    ## feature map histogram
    for batch, batch_data in enumerate(trainLoader):
        inputs = batch_data[0]['data'].to(device)
        break
    fm_all = get_model_featuremap(model,inputs)
    for index in range(len(fm_all)):
        if index==0:
            select_fm = fm_all[index]
            activation= quan_activation(select_fm, a_s[index], a_beta[index],a_bit[index], unilateral=False)
        else:
            # if a_bit[index] == 1:
            #     select_fm = fm_all[index]
            # else:
            #     mask = fm_all[index] > 0
            #     select_fm = torch.masked_select(fm_all[index], mask)
            select_fm = fm_all[index]
            activation= quan_activation(select_fm, a_s[index], a_beta[index],a_bit[index], unilateral=True)

        fparam = select_fm.view(-1).detach().cpu().numpy()
        quant_param=activation.view(-1).detach().cpu().numpy()
        ax1[index].set_title(come[index])
        ax1[index].hist(fparam,bins=200,color='steelblue', density=True)
        qaun_ax  = ax1[index].twinx()
        qaun_ax.hist(quant_param,bins=200,color='red',density=True)
        # ax1[index].set_xticks([])
        # ax1[index].set_yticks([])
        # qaun_ax.set_xticks([])
        # qaun_ax.set_yticks([])

    fig1.tight_layout(pad=0.2, w_pad=-2.5, h_pad=1.0)
    # plt.tight_layout()
    plt.show()
    fig1.savefig('resnet18_3.pdf',dpi=600,format='pdf')
    print('featuremap hist saved!')

if __name__ == '__main__':
    main()