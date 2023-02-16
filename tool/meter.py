# -*- coding:utf-8  -*-

import numpy as np
import torch
import torch.nn as nn
from tool.thop import profile

def get_params_flops(model,input_size=32):
    # Model flops and params
    input = torch.randn(1, 3, input_size, input_size)
    _, _, model_all = profile(model, inputs=(input, ))

    cnt = 0
    flops = []
    params = []
    conv_num = 0
    fc_num = 0
    total_params = 0
    total_flops = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        if (isinstance(m, nn.Conv2d)):
            print("CONV-%s Params=%d Flops=%3f" % (str(cnt), m.total_params, m.total_ops))
            flops.append(m.total_ops)
            params.append(m.total_params)
            total_params += m.total_params
            total_flops += m.total_ops
            cnt = cnt + 1
            conv_num = conv_num + 1
        elif (isinstance(m, nn.Linear)):
            print("FC-%s Params=%d  Flops=%3f" % (str(cnt), m.total_params, m.total_ops))
            flops.append(m.total_ops)
            params.append(m.total_params)
            total_params += m.total_params
            total_flops += m.total_ops
            cnt = cnt + 1
            fc_num = fc_num + 1
    print("Total Params = %d  |  Total Flops = %d" % (total_params, total_flops))

    return flops,params,total_params,total_flops,conv_num,fc_num

def calc_model_flops(model, input_size, mul_add=False):
    hook_list = []
    module_flops = []

    def conv_hook(self, input, output):
        output_channels, output_height, output_width = output[0].size()
        bias_ops = 1 if self.bias is not None else 0
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        flops = (kernel_ops * (2 if mul_add else 1) + bias_ops) * output_channels * output_height * output_width
        module_flops.append(flops)

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement() * (2 if mul_add else 1)
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        module_flops.append(flops)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(2, 3, input_size, input_size).to(next(model.parameters()).device)
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return  module_flops


def calc_model_parameters_all(model):
    total_params = 0

    params = list(model.parameters())
    for param in params:
        cnt = 1
        for d in param.size():
            cnt *= d
        total_params += cnt

    return round(total_params / 1e6, 2)

def calc_model_parameters(model):
    params = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weights_num, weights_channel, weights_height, weights_width = m.weight.data.size()
            param = weights_num*weights_channel*weights_height*weights_width
            params.append(param)
        elif isinstance(m, nn.Linear):
            weights_in, weights_out = m.weight.data.size()
            param = weights_in*weights_out
            # print(param)
            params.append(param)

    # return round(total_params / 1e6, 2)
    return params

def calc_model_featuremap(model, input_size):
    hook_list = []
    module_featuremap = []

    def conv_hook(self, input, output):
        # print(input[0].size())
        batch, input_channels, input_height, input_width = input[0].size()
        featuremap = input_channels * input_height * input_width
        module_featuremap.append(featuremap)

    def linear_hook(self, input, output):
        # print(input[0].size())
        batch,input_channels = input[0].size()
        featuremap = input_channels 
        module_featuremap.append(featuremap)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(2, 3, input_size, input_size).to(next(model.parameters()).device)
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return module_featuremap
