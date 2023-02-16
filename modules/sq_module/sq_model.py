import torch
import numpy as np
import torch.nn as nn
import copy as copy
from .SQConv import *
from .SQLinear import *
from .activation import *
from .DyReLU import *

search_cnt=[0]
def convert_to_sqconv(model,w_bit,a_bit,pruning_rate):
    global search_cnt
    for name, module in(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_sqconv(module,w_bit,a_bit,pruning_rate)
 
        if type(module) == nn.Conv2d:
            sq_conv2d = SQConv(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias is not None,
                               w_bit = w_bit[search_cnt[0]], a_bit = a_bit[search_cnt[0]], pruning_rate=pruning_rate[search_cnt[0]], QInput = True, bSetQ = True)
            search_cnt[0] += 1
            channel = module.out_channels
            model._modules[name] = sq_conv2d

        elif type(module) == nn.Linear:
            sq_linear = SQLinear(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None,
                                 w_bit = w_bit[search_cnt[0]], a_bit = a_bit[search_cnt[0]], pruning_rate=pruning_rate[search_cnt[0]], QInput = True, bSetQ = True)
            search_cnt[0] += 1           
            model._modules[name] = sq_linear

        elif type(module) == nn.ReLU or type(module) == nn.ReLU6:
            # print(a_bit[search_cnt[0]])
            # activation = PRelu()
            # activation = PACT()
            # activation = nn.LeakyReLU()
            # activation = DyReLUB(channel)
            activation = Scale_HardTanh(channel, bit=a_bit[search_cnt[0]], per_channel=False)
            # activation = Scale_HardTanh()
            model._modules[name] = activation

            # try:
            #     if  a_bit[search_cnt[0]] == 1:
            #         activation = Scale_HardTanh(bit=a_bit[search_cnt[0]])
            #         model._modules[name] = activation
            #     elif a_bit[search_cnt[0]] <= 4:
            #         activation = Scale_PACT()
            #         model._modules[name] = activation
            #     else:
            #         activation = module
            #         model._modules[name] = activation
            # except IndexError:
            #     # print("The Last Activation")
            #     activation = module
            #     model._modules[name] = activation     
    return model