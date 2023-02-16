## this is a tool for feature map and weights histogram
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import time
from importlib import import_module
import copy
import sys
sys.path.append('..')
from utils.common import *
from data import cifar10
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
from module.binary.binary_model import *
import models.cifar10 as models
# from torchsummary import summary
import matplotlib.pyplot as plt
from module.binary import BinaryConv2d

def quan_activation(inputs, ):


def quan_weights(inputs, ):


def fp_hist(model, inputs, index):
    param = []

    data = get_model_featuremap(model, inputs)

    cnt = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            if cnt == index:
                a_s = m.a_s.data
                a_beta = m.a_beta.data
            cnt += 1
    param = data[index]


    print(param.size())
    param = param.view(-1)
    param = param.detach().cpu().numpy()

    fig=plt.figure()
    ax=fig.add_subplot(111) 
    ax.set_title('feature map')
    ax.hist(param,bins=100,color='red',density=False)
    plt.savefig('/home/lab611/workspace/lvjunhuan/core_replace_prj/picture/feature-6')



def wt_hist(model,index):
    param = []

    data = get_params_model(model)

    cnt = 0
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            if cnt == index:
                w_s = m.w_s.data
            cnt += 1
    param = data[index]


    param = data[83]
    print(param.size())
    param = param.view(-1)
    param = param.detach().cpu().numpy()

    fig=plt.figure()
    ax=fig.add_subplot(111) 
    ax.set_title('weight')
    ax.hist(param,bins=100,color='red',density=False)
    plt.savefig('/home/lab611/workspace/lvjunhuan/core_replace_prj/picture/weight-1')


# def come_hist(data):
#     param = convert(data)
#     fig=plt.figure()
#     ax=fig.add_subplot(111) 
#     ax.set_title('feature map')
#     ax.hist(param,bins=100,color='blue',density=False)
#     plt.savefig('../picture/feature-1')

# def convert(data)
#     param = data[0]
#     param = param.view(-1)
#     param = param.detach().cpu().numpy()
#     return param
 
# def come_hist(data):
#     fig=plt.figure()
#     ax=fig.add_subplot(111) 
#     ax.set_title('feature map')
#     ax.hist(data,bins=15,color='blue',density=False)
#     plt.savefig('/home/lab611/workspace/lvjunhuan/core_replace_prj/picture/feature-3')

    #param = data[0]
 #   print(param.size())
#     axparam = data.view(-1)
#    param = param.detach().cpu().numpy()