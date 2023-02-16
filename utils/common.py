from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os
# import cv2
import sys
import copy
import torch
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as torch_func

# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='None', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_checkpoint.pt'
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def get_pruning_rate(pruning_rate):
    import re

    cprate_str = pruning_rate
    print(cprate_str)
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    return cprate

def get_params_model(model):
    params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            params.append(m.weight.data)
            if m.bias is not None:
                params.append(m.bias.data)
        elif isinstance(m, nn.Linear):
            params.append(m.weight.data)
            if m.bias is not None:
                params.append(m.bias.data)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            params.append(m.running_mean.data)
            params.append(m.running_var.data)
            if m.weight is not None:
                params.append(m.weight.data)
            if m.bias is not None:
                params.append(m.bias.data)
    return params

def get_params_model_dict(model):
    params = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            params[name + ".weight"] = m.weight.data
            if m.bias is not None:
                params[name + ".bias"] = m.bias.data
        elif isinstance(m, nn.Linear):
            params[name + ".weight"] = m.weight.data
            if m.bias is not None:
                params[name + ".bias"] = m.bias.data
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            params[name + ".running_mean"] = m.running_mean.data
            params[name + ".running_var"] = m.running_var.data
            if m.weight is not None:
                params[name + ".weight"] = m.weight.data
            if m.bias is not None:
                params[name + ".bias"] = m.bias.data
    for k in params.keys():
        print(k)
    return params

def load_params_model(model,params): # 把merge BN后的参数加载进修改后的模型
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = params[cnt]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, nn.Linear):
            m.weight.data = params[cnt]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
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
    return model

def convert_keys(model, baseline):
    '''
    rename the baseline's key to model's name
    e.g.
        baseline_ckpt = torch.load(args.bsealine, map_location=device)
        model.load_state_dict(convert_keys(model, baseline_ckpt))
    '''
    from collections import OrderedDict

    baseline_state_dict = OrderedDict()
    model_key = list(model.state_dict().keys())
    # baseline_key = list(baseline['state_dict'].keys())
    ### 这里修改：根据打印出来的 加载模型参数orderdict对象进行修改 打印出的内容显示该对象没有'state_dict'字段 可以直接进行
    baseline_key = list(baseline.keys())
    if(len(model_key)!=len(baseline_key)):
        print("ERROR: The model and the baseline DO NOT MATCH")
        pdb.set_trace()
        exit()
    else:
        for i in range(len(model_key)):
            # baseline_state_dict[model_key[i]] = baseline['state_dict'][baseline_key[i]]
            baseline_state_dict[model_key[i]] = baseline[baseline_key[i]]
    return baseline_state_dict

def const(num=0.5):
    return num


def linear(epoch,EP):
    return 1 / (EP - 1) * (epoch - 1)


def log(epoch,EP):
    return math.log(epoch+1, EP)


def exp(epoch,EP):
    return 2 ** (epoch / 5) / (2 ** ((EP / 5) - 1)) / 2


def step(epoch,EP):
    if epoch < EP / 4:
        return 0
    if epoch < EP / 2:
        return 1 / 3
    if epoch < EP * 3 / 4:
        return 2 / 3
    else:
        return 1

def sig(epoch,EP):
    scale = 5
    return 1 / (1 + np.exp(-(epoch / scale - EP / (scale * 2))))

def get_layer_types(feat_layers):
    conv_layers = []
    for layer in feat_layers:
        if not isinstance(layer, nn.Linear) and not isinstance(layer, nn.BatchNorm1d):
            conv_layers.append(layer)
    return conv_layers

def get_net_info(net, data_size=32, as_module=False):
    device = next(net.parameters()).device
    if isinstance(net, nn.DataParallel):
        net = net.module
    layers = list(net.children())
    feat_layers = get_layer_types(layers)
    linear = layers[-1]
    channels = []
    input_size = [[3, data_size, data_size]]
    x = [torch.rand(2, *in_size) for in_size in input_size]
    x = torch.Tensor(*x).to(device)
    for layer in feat_layers:
        x = layer(x)
        channels.append(x.shape)
    if as_module:
        return nn.ModuleList(feat_layers), linear, channels
    return feat_layers, linear, channels

def get_layers_feats(x, layers):
    layer_feats = []
    out = x
    for layer in layers:
        out = layer(out)
        if isinstance(layer, nn.Sequential):
            layer_feats.append(out)
    return layer_feats
