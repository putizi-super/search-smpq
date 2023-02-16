import torch
import torch.nn as nn
import torch.nn.init as init
import sys
sys.path.append('../../')
from models.imagenet.reactnet import HardBinaryConv


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


def _merge_bn(conv_module, bn_module):
    alpha = None
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    if (isinstance(conv_module, nn.Conv2d)) or isinstance(conv_module, nn.Linear):
        w_view = (conv_module.out_channels, 1, 1, 1)
        if bn_module.affine:
            weight = w * (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
    else:
        weight = w
        w_view = (conv_module.shape[0], 1, 1, 1)
        if bn_module.affine:
            alpha = (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            alpha = 1 / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
    return weight, bias, alpha


def merge_bn_into_conv(conv_module, bn_module):
    w, b, alpha= _merge_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    if (isinstance(conv_module, nn.Conv2d)) or isinstance(conv_module, nn.Linear):
        conv_module.weight.data = w
    else:
        conv_module.weight.data = w
        conv_module.alpha = alpha
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2 -bn_module.eps


def reset_bn(module: nn.BatchNorm2d):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1-module.eps)
        # we do not reset numer of tracked batches here
        # self.num_batches_tracked.zero_()
    if module.affine:
        init.ones_(module.weight)
        init.zeros_(module.bias)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear) or isinstance(m, HardBinaryConv)

def search_merge_and_remove_bn(model):
    model.eval()
    prev = None
    prev1 = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            merge_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
            prev = m
        elif is_absorbing(m):
            if is_absorbing(prev):
                prev1 = m
            else:
                prev = m
        elif is_bn(m) and is_bn(prev):
            merge_bn_into_conv(prev1, m)
        else:
            prev = search_merge_and_remove_bn(m)
    return prev


def search_merge_and_reset_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            merge_bn_into_conv(prev, m)
            # reset_bn(m)
        else:
            search_merge_and_reset_bn(m)
        prev = m

