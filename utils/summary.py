from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import collections
from utils import *

def summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1
           
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size))
    # create properties
    summary = collections.OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    return summary


def summary2(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = '%s-%i' % (class_name, module_idx+1)
            if class_name=='Conv2d':
                summary[m_key] = collections.OrderedDict()
                batch_size, input_channles, input_height, input_width = input[0].size()
                output_channels, output_height, output_width = output[0].size()
                # [kernel_y, kernel_x] = module.kernel_size
                wt_params = module.kernel_size[0]*module.kernel_size[1]*input_channles*output_channels/module.groups
                # wt_params = module.kernel_size[0]*module.kernel_size[1]*module.kernel_size[2]*module.kernel_size[3]

                wt_mops = wt_params*output_height*output_width/1e6
                summary[m_key]['params'] = wt_params
                summary[m_key]['mops'] = wt_mops
            elif class_name=='Linear':
                summary[m_key] = collections.OrderedDict()
                batch_size, input_channles = input[0].size()
                output_channels = output[0].size(0)
                # [kernel_y, kernel_x] = module.kernel_size

                wt_params = input_channles*output_channels
                wt_mops = wt_params/1e6
                summary[m_key]['params'] = wt_params
                summary[m_key]['mops'] = wt_mops

           
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size))
    # create properties
    summary = collections.OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    return summary

def display(summary, prun_ratio=[]):
    key_list = []
    params_list = []
    mops_list = []
    for key,value in summary.items():
        key_list.append(key)
        for vk,vv in value.items():
            if(vk=='params'):
                params_list.append(vv)
            if(vk=='mops'):
                mops_list.append(vv)

    print("\n===========================================================================================================")
    print('conv               prun_ratio             params (res/orig)          MFLOPs (res/orig)')
 
    i=0
    sum_p_prun = 0
    sum_p_ori = 0
    sum_ops_prun = 0
    sum_ops_ori = 0
    for name in key_list:

        # summary_str = name + "             " + \
        # str(round(100*prun_ratio[i], 2)) + "             " +  \
        # str(round(params_list[i]*(1-prun_ratio[i]))) + "/" + \
        # str(params_list[i]) + "             " + \
        # str(round(mops_list[i]*(1-prun_ratio[i]), 2)) + "/" + \
        # str(round(mops_list[i], 2))

        ratio = round(100*(1-prun_ratio[i]), 2)
        p_prun = round(params_list[i]*prun_ratio[i])
        p_ori = params_list[i]
        ops_prun = round(mops_list[i]*prun_ratio[i], 3)
        ops_ori = round(mops_list[i], 3)

        sum_p_prun += p_prun
        sum_p_ori += p_ori
        sum_ops_prun += ops_prun
        sum_ops_ori += ops_ori

        print(str(name).ljust(20), end='')
        print((str(ratio)+'%').ljust(20),  end='')
        print((str(p_prun).ljust(7)+'/'),  end='')
        print(str(p_ori).ljust(20),  end='')
        print((str(ops_prun).ljust(7)+'/'),  end='')
        print(str(ops_ori))
		
        i = i+1

    print(str('total(res)').ljust(20), end='')
    print((str('')).ljust(20),  end='')
    print((str(sum_p_prun).ljust(7)+'/'),  end='')
    print(str(sum_p_ori).ljust(8),  end='')
    print(('('+str(round(100.0*sum_p_prun/sum_p_ori, 2))+'%)').ljust(12),  end='')
    print((str(round(sum_ops_prun, 3)).ljust(6)+'/'),  end='')
    print(str(round(sum_ops_ori,3)).ljust(7), end='')
    print('('+str(round(100.0*sum_ops_prun/sum_ops_ori, 2))+'%)')
    print((str('total(pruned)')).ljust(20),  end='')
    print((str('')).ljust(20),  end='')
    print(str().ljust(7),  end='')
    print(str().ljust(8),  end='')
    print((str(round(100.0*(1-sum_p_prun/sum_p_ori), 2))+'%').ljust(12),  end='')
    print(str().ljust(6),  end='')
    print(str().ljust(7), end='')
    print(str(round(100.0*(1-sum_ops_prun/sum_ops_ori), 2))+'%')
    print("==========================================================================================================\n")


def summary_display(model, input_size, prun_ratio):
    summary = summary2(input_size, model)
    display(summary, prun_ratio)

# # a_net = A()
# a_net = LeNet5()
# a = summary2((1,28,28), a_net)
# desplay(a)
# # print(a)


