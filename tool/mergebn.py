import torch
import torch.nn as nn
from modules.sq_module.sq_model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys

def fillbias(model):  # 这个函数不太理解用途，虽然上面标注给原始模型加上偏置，但是自我感觉：将model的模块name进行更新
    for name, module in(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = fillbias(module)
        elif type(module) == nn.Conv2d:
            conv = nn.Conv2d(in_channels=module.in_channels, out_channels=module.out_channels, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, bias=True)
            model._modules[name] = conv
    return model

def convert_to_mergebn(model):  # 去掉Bn层
    for name, module in(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_to_mergebn(module)
        elif type(module) == nn.BatchNorm2d:
            bn = nn.Sequential()
            model._modules[name] = bn
    return model


def get_params_mergebn(model):
    cnt = 0                     # 统计 params中元素数量
    params = []
    for name, m in model.named_modules():   
        if isinstance(m, nn.Conv2d):   # m 卷积层   将权重、偏置（若没有置0)，加入params
            params.append(m.weight.data)
            cnt += 1                    
            if m.bias is not None:
                params.append(m.bias.data)
                cnt += 1
            else:
                params.append(torch.zeros(m.weight.data.size(0)).to(device))
                cnt += 1
        elif isinstance(m, nn.Linear):  # m 线性层 将权重、偏置（若没有不加），加入params
            params.append(m.weight.data)
            cnt += 1
            if m.bias is not None:
                params.append(m.bias.data)
                cnt += 1
        elif isinstance(m, nn.BatchNorm1d):  # m batchnorm1d层 将 平均值、方差、权重、偏置（若没有则不加入），加入params 
            params.append(m.running_mean.data)
            cnt += 1
            params.append(m.running_var.data)
            cnt += 1 
            if m.weight is not None:
                params.append(m.weight.data)
                cnt += 1
            if m.bias is not None:
                params.append(m.bias.data)
                cnt += 1
        elif isinstance(m, nn.BatchNorm2d):  # m batchnorm2d层  （这个地方不理解：为什么这个是将原有的params进行修改）
            for i in range(params[cnt-2].size(0)):
                params[cnt-2][i] = params[cnt-2][i] * m.weight.data[i]/torch.sqrt(m.running_var.data[i] + m.eps)
                params[cnt-1][i] = m.bias.data[i] - m.running_mean.data[i] * m.weight.data[i]/torch.sqrt(m.running_var.data[i] + m.eps)
    return params

def get_params_mergebn_sq(model):
    cnt = 0
    params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            params.append(m.a_s.data)
            cnt += 1
            params.append(m.a_beta.data)
            cnt += 1
            params.append(m.w_s.data)
            cnt += 1
            params.append(m.weight.data)
            cnt += 1
            if m.bias is not None:
                params.append(m.bias.data)
                cnt += 1
            else:
                params.append(torch.zeros(m.weight.data.size(0)).to(device))
                cnt += 1
        elif isinstance(m, nn.Linear):
            # for pname, param in m.named_parameters():
            #     print(pname)
            params.append(m.a_s.data)
            cnt += 1
            params.append(m.a_beta.data)
            cnt += 1
            params.append(m.w_s.data)
            cnt += 1
            params.append(m.b_s.data)
            cnt += 1
            params.append(m.weight.data)
            cnt += 1
            if m.bias is not None:
                params.append(m.bias.data)
                cnt += 1
        elif isinstance(m, nn.BatchNorm1d):
            params.append(m.running_mean.data)
            cnt += 1
            params.append(m.running_var.data)
            cnt += 1
            if m.weight is not None:
                params.append(m.weight.data)
                cnt += 1
            if m.bias is not None:
                params.append(m.bias.data)
                cnt += 1
        elif isinstance(m, nn.BatchNorm2d):
            A = torch.ones(params[cnt-2].size(0), 1, 1, 1).to(device)
            for i in range(params[cnt-2].size(0)):
                params[cnt-2][i] = params[cnt-2][i] * m.weight.data[i]/torch.sqrt(m.running_var.data[i] + m.eps)
                params[cnt-1][i] = m.bias.data[i] - m.running_mean.data[i] * m.weight.data[i]/torch.sqrt(m.running_var.data[i] + m.eps)
                A[i][0][0][0] = m.weight.data[i]/torch.sqrt(m.running_var.data[i] + m.eps)*params[cnt-3]
            params[cnt-3] = A
        elif isinstance(m, Scale_HardTanh):
            params.append(m.b.data)
            cnt += 1
            params.append(m.k.data)
            cnt += 1
        elif isinstance(m, Scale_PACT):
            params.append(m.b.data)
            cnt += 1
            params.append(m.k.data)
            cnt += 1
            params.append(m.alpha.data)
            cnt += 1
    return params

def load_params_mergebn_sq(model,params):
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.a_s.data = params[cnt]
            cnt += 1
            m.a_beta.data = params[cnt]
            cnt += 1
            m.w_s.data = params[cnt]
            cnt += 1
            m.weight.data = params[cnt]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, nn.Linear):
            m.a_s.data = params[cnt]
            cnt += 1
            m.a_beta.data = params[cnt]
            cnt += 1
            m.w_s.data = params[cnt]
            cnt += 1
            m.b_s.data = params[cnt]
            cnt += 1
            m.weight.data = params[cnt]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # for pname, param in m.named_parameters():
            #     print(pname)
            # sys.exit()
            m.running_mean.data = params[cnt]
            cnt += 1
            m.running_var.data = params[cnt]
            cnt += 1
            if m.weight is not None:
                m.weight.data = params[cnt]
                cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, Scale_HardTanh):
            m.b.data = params[cnt]
            cnt += 1
            m.k.data = params[cnt]
            cnt += 1
        elif isinstance(m, Scale_PACT):
            m.b.data = params[cnt]
            cnt += 1
            m.k.data = params[cnt]
            cnt += 1
            m.alpha.data = params[cnt]
            cnt += 1
    return model


if __name__ == "__main__":
    params = get_params_mergebn(model)
    model = fillbias(model)
    model = convert_to_mergebn(model)
    model = utils.load_params_model(model,params)