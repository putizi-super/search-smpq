from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os
import sys
import copy
import torch
import logging
import torch.nn as nn

def get_idx(params, res_num, method='l2'):
    if method.lower() == 'l2':
        l2 = torch.sqrt(torch.sum(params.pow(2), dim=[-1, -2, -3]))
        idx = torch.sort(l2, descending=True)[1][:res_num]
    elif method.lower() == "l1":
        l1 = torch.sum(torch.abs(params), dim=[-1, -2, -3])
        idx = torch.sort(l1, descending=True)[1][:res_num]
    elif method.lower() == "gm":
        distance = []
        for fi in params:
            dist = torch.sum([torch.sqrt(torch.sum((fi - fj).pow(2))) for fj in params]).item()
            distance += [dist]
        idx = torch.sort(distance, descending=True)[1][:res_num]
    return torch.sort(idx)[0]

def load_params_pruned_resnet_l2(pruned_model, origin_state_dict):
    # cfg = {
    #     "resnet20": [3, 3, 3],
    #     "resnet32": [5, 5, 5],
    #     "resnet44": [7, 7, 7],
    #     'resnet56': [9, 9, 9],
    #     'resnet110': [18, 18, 18],
    # }
    pruned_state_dict = pruned_model.state_dict()
    is_preserved = False
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("****************************** Conv2d ****************************************")
            pruned_weight = pruned_state_dict[name + ".weight"]
            origin_weight = origin_state_dict[name + ".weight"]
            # print(pruned_weight.shape)
            
            if pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                print("===> channel&Filter")
                # print(origin_weight)
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                origin_weight = origin_weight[:, preserved_idx]
                print(preserved_idx)
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_weight = origin_weight[idx]
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                print("---")
                print(idx)
                # print(pruned_weight)
                print()
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                print("===> Filter")
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                is_preserved = True
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
                print(idx)
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                print("===> channel")
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                pruned_state_dict[name + ".weight"] = origin_weight[:, preserved_idx]
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
                print(preserved_idx)
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                print("===> nothing")

                pruned_state_dict[name + ".weight"] = origin_weight
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            
            # print(origin_weight.shape)
            if is_preserved:
                preserved_idx = copy.deepcopy(idx)
        
        if isinstance(module, nn.Linear):
            pruned_weight = pruned_state_dict[name + ".weight"]
            origin_weight = origin_state_dict[name + ".weight"]
            if pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                origin_weight = origin_weight[:, preserved_idx]
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                pruned_state_dict[name + ".weight"] = origin_weight[:, preserved_idx]
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            elif pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                is_preserved = True
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                print("Linner")
                pruned_state_dict[name + ".weight"] = origin_weight
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            if is_preserved:
                preserved_idx = copy.deepcopy(idx)
        
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            print("****************************** BN    ****************************************")
            bn_mean_name = name + '.running_mean'
            bn_var_name = name + '.running_var'
            if pruned_state_dict[bn_mean_name].size(0) != origin_state_dict[bn_mean_name].size(0):
                print(preserved_idx)
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                pruned_state_dict[bn_mean_name] = origin_state_dict[bn_mean_name][preserved_idx]
                pruned_state_dict[bn_var_name] = origin_state_dict[bn_var_name][preserved_idx]
                if module.weight is not None:
                    bn_weight_name = name + '.weight'
                    pruned_state_dict[bn_weight_name] = origin_state_dict[bn_weight_name][preserved_idx]
                if module.bias is not None:
                    bn_bias_name = name + '.bias'
                    pruned_state_dict[bn_bias_name] = origin_state_dict[bn_bias_name][preserved_idx]
            else:
                pruned_state_dict[bn_mean_name] = origin_state_dict[bn_mean_name]
                pruned_state_dict[bn_var_name] = origin_state_dict[bn_var_name]
                if module.weight is not None:
                    bn_weight_name = name + '.weight'
                    pruned_state_dict[bn_weight_name] = origin_state_dict[bn_weight_name]
                if module.bias is not None:
                    bn_bias_name = name + '.bias'
                    pruned_state_dict[bn_bias_name] = origin_state_dict[bn_bias_name]
    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model

def load_params_pruned_resnet(pruned_model, origin_state_dict):
    pruned_state_dict = pruned_model.state_dict()
    is_preserved = False
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            pruned_weight = pruned_state_dict[name + ".weight"]
            origin_weight = origin_state_dict[name + ".weight"]
            # print(pruned_weight.shape)
            
            if pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                # print(origin_weight)
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                origin_weight = origin_weight[:, preserved_idx]
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_weight = origin_weight[idx]
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                is_preserved = True
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                pruned_state_dict[name + ".weight"] = origin_weight[:, preserved_idx]
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                pruned_state_dict[name + ".weight"] = origin_weight
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            
            if is_preserved:
                preserved_idx = copy.deepcopy(idx)
        
        if isinstance(module, nn.Linear):
            pruned_weight = pruned_state_dict[name + ".weight"]
            origin_weight = origin_state_dict[name + ".weight"]
            if pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                origin_weight = origin_weight[:, preserved_idx]
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) != origin_weight.size(1):
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                pruned_state_dict[name + ".weight"] = origin_weight[:, preserved_idx]
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            elif pruned_weight.size(0) != origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                idx = get_idx(origin_weight, pruned_weight.size(0))
                pruned_state_dict[name + ".weight"] = origin_weight[idx]
                is_preserved = True
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"][idx]
            elif pruned_weight.size(0) == origin_weight.size(0) and pruned_weight.size(1) == origin_weight.size(1):
                pruned_state_dict[name + ".weight"] = origin_weight
                is_preserved = False
                if module.bias is not None:
                    pruned_state_dict[name + ".bias"] = origin_state_dict[name + ".bias"]
            if is_preserved:
                preserved_idx = copy.deepcopy(idx)
        
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_mean_name = name + '.running_mean'
            bn_var_name = name + '.running_var'
            if pruned_state_dict[bn_mean_name].size(0) != origin_state_dict[bn_mean_name].size(0):
                if not is_preserved:
                    raise "sth wrong with the settings of `is_preserved`"
                pruned_state_dict[bn_mean_name] = origin_state_dict[bn_mean_name][preserved_idx]
                pruned_state_dict[bn_var_name] = origin_state_dict[bn_var_name][preserved_idx]
                if module.weight is not None:
                    bn_weight_name = name + '.weight'
                    pruned_state_dict[bn_weight_name] = origin_state_dict[bn_weight_name][preserved_idx]
                if module.bias is not None:
                    bn_bias_name = name + '.bias'
                    pruned_state_dict[bn_bias_name] = origin_state_dict[bn_bias_name][preserved_idx]
            else:
                pruned_state_dict[bn_mean_name] = origin_state_dict[bn_mean_name]
                pruned_state_dict[bn_var_name] = origin_state_dict[bn_var_name]
                if module.weight is not None:
                    bn_weight_name = name + '.weight'
                    pruned_state_dict[bn_weight_name] = origin_state_dict[bn_weight_name]
                if module.bias is not None:
                    bn_bias_name = name + '.bias'
                    pruned_state_dict[bn_bias_name] = origin_state_dict[bn_bias_name]
    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model
    return pruned_model

def load_params_model_fp(model, params):
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # m.weight.data = params[cnt]
            # print(m.weight.data.size())
            # print(params[cnt][0:m.weight.data.size(0),0:m.weight.data.size(1),:,:].size())
            m.weight.data = params[cnt][0:m.weight.data.size(0), 0:m.weight.data.size(1), :, :]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt][0:m.bias.data.size(0)]
                cnt += 1
        elif isinstance(m, nn.Linear):
            # print(m.weight.data.size())
            m.weight.data = params[cnt][0:m.weight.data.size(0),0:m.weight.data.size(1)]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt][0:m.bias.data.size(0)]
                cnt += 1
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # print(m.running_mean.data.size())
            # print(m.running_var.data.size())
            m.running_mean.data = params[cnt][0:m.running_mean.data.size(0)]
            cnt += 1
            m.running_var.data = params[cnt][0:m.running_var.data.size(0)]
            cnt += 1
            if m.weight is not None:
                # print(m.weight.data.size())
                m.weight.data = params[cnt][0:m.weight.data.size(0)]
                cnt += 1
            if m.bias is not None:
                # print(m.bias.data.size())
                m.bias.data = params[cnt][0:m.bias.data.size(0)]
                cnt += 1
    return model

def load_params_model_fp_resize(model,params):
    import cv2
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # m.weight.data = params[cnt]
            # print(m.weight.data.size())
            # print(params[cnt][0:m.weight.data.size(0),0:m.weight.data.size(1),:,:].size())
            A = params[cnt].view(params[cnt].size(0),-1)
            B = torch.from_numpy(cv2.resize(A.cpu().numpy(),(m.weight.data.size(1)*m.weight.data.size(2)*m.weight.data.size(3),m.weight.data.size(0)),interpolation=cv2.INTER_CUBIC)).to(device)
            m.weight.data = B.view(m.weight.data.size())
            cnt += 1
            if m.bias is not None:
                A = torch.unsqueeze(params[cnt],1)
                B = torch.squeeze(torch.from_numpy(cv2.resize(A.cpu().numpy(),(1, m.bias.data.size(0)),interpolation=cv2.INTER_CUBIC))).to(device)
                m.bias.data = B
                cnt += 1
        elif isinstance(m, nn.Linear):
            A = params[cnt].view(params[cnt].size(0),-1)
            B = torch.from_numpy(cv2.resize(A.cpu().numpy(),(m.weight.data.size(1), m.weight.data.size(0)),interpolation=cv2.INTER_CUBIC)).to(device)
            m.weight.data = B.view(m.weight.data.size())
            cnt += 1
            if m.bias is not None:
                A = torch.unsqueeze(params[cnt],1)
                B = torch.squeeze(torch.from_numpy(cv2.resize(A.cpu().numpy(),(1, m.bias.data.size(0)),interpolation=cv2.INTER_CUBIC))).to(device)
                m.bias.data = B
                cnt += 1
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # print(m.running_mean.data.size())
            # print(m.running_var.data.size())

            A = torch.unsqueeze(params[cnt],1)
            B = torch.squeeze(torch.from_numpy(cv2.resize(A.cpu().numpy(),(1, m.running_mean.data.size(0)),interpolation=cv2.INTER_CUBIC))).to(device)
            m.running_mean.data = B
            cnt += 1
            A = torch.unsqueeze(params[cnt],1)
            B = torch.squeeze(torch.from_numpy(cv2.resize(A.cpu().numpy(),(1, m.running_var.data.size(0)),interpolation=cv2.INTER_CUBIC))).to(device)
            m.running_var.data = B
            cnt += 1
            if m.weight is not None:
                # print(m.weight.data.size())
                A = torch.unsqueeze(params[cnt],1)
                B = torch.squeeze(torch.from_numpy(cv2.resize(A.cpu().numpy(),(1, m.weight.data.size(0)),interpolation=cv2.INTER_CUBIC))).to(device)
                m.weight.data = B
                cnt += 1
            if m.bias is not None:
                # print(m.bias.data.size())
                A = torch.unsqueeze(params[cnt],1)
                B = torch.squeeze(torch.from_numpy(cv2.resize(A.cpu().numpy(),(1, m.bias.data.size(0)),interpolation=cv2.INTER_CUBIC))).to(device)
                m.bias.data = B
                cnt += 1
    return model
