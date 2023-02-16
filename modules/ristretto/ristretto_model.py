import torch
import numpy as np
import torch.nn as nn
import copy as copy
from .RistrettoConv import *
from .RistrettoLinear import *

ristretto_cnt = [0]
def convert_to_ristretto(model,w_bit,a_bit,pruning_rate):
    global ristretto_cnt
    for name, module in(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_to_ristretto(model=module,w_bit=w_bit,a_bit=a_bit,pruning_rate=pruning_rate)
 
        if type(module) == nn.Conv2d:
            ristretto_conv2d = RistrettoConv(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias is not None,
                                      w_bit = w_bit[ristretto_cnt[0]], a_bit=a_bit[ristretto_cnt[0]], pruning_rate=pruning_rate[ristretto_cnt[0]],QInput = True, bSetQ = True)
            ristretto_cnt[0] += 1
            model._modules[name] = ristretto_conv2d

        elif type(module) == nn.Linear:
            ristretto_linear = RistrettoLinear(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, 
                                        w_bit = w_bit[ristretto_cnt[0]], a_bit=a_bit[ristretto_cnt[0]], pruning_rate=pruning_rate[ristretto_cnt[0]], QInput = True, bSetQ = True)
            ristretto_cnt[0] += 1
            model._modules[name] = ristretto_linear

    return model
