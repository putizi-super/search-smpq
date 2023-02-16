import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
# from sklearn.cluster import KMeans
from .base_uaq import UniformAffineQuantizer
from .base_lsq import LSQQuantizer
from .base_ristretto import RistrettoQuantizer

class StraightThrough(nn.Module): ### 此函数的作用
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear],  p: float = 2.4, weight_quant_params: dict = {}, bias_quant_params: dict = {},
                 act_quant_params: dict = {}, use_in_quant: bool = False, disable_act_quant: bool = False, quantizer: str = 'ristretto'):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        self.ada_weight = None
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.save_params = False
        self.cluster_init = True
        self.per_channel = True
        self.use_in_quant = use_in_quant
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        if quantizer == 'lsq':
            self.weight_quantizer = LSQQuantizer(**weight_quant_params)
            self.bias_quantizer = LSQQuantizer(**bias_quant_params)
            self.in_act_quantizer = LSQQuantizer(**act_quant_params)
            self.out_act_quantizer = LSQQuantizer(**act_quant_params)
        elif quantizer == 'ristretto':
            self.weight_quantizer = RistrettoQuantizer(p, **weight_quant_params)
            self.bias_quantizer = RistrettoQuantizer(p, **bias_quant_params)
            self.in_act_quantizer = RistrettoQuantizer(p, **act_quant_params)
            self.out_act_quantizer = RistrettoQuantizer(p, **act_quant_params)
        elif quantizer == 'uaq':
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
            self.bias_quantizer = UniformAffineQuantizer(**bias_quant_params)
            self.in_act_quantizer = UniformAffineQuantizer(**act_quant_params)
            self.out_act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            raise NotImplementedError
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

    def forward(self, input: torch.Tensor):
        # if self.cluster_init:
        #     if self.per_channel:
        #         for i in range(self.weight.data.shape[0]):
        #             self.weight.data[i] = self.init_cluster(self.weight.data[i], 10)
        #     else:
        #         self.weight.data = self.init_cluster(self.weight.data, 10)
        #     print("Cluster Finish!")
        #     self.cluster_init = False
        
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias_quantizer(self.bias)
            # bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
            # weight = self.weight
            # bias = self.bias

        # quantitative input
        if self.use_in_quant:
        # if self.use_act_quant:
            input = self.in_act_quantizer(input)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # save quantized weight
        if self.save_params:
            self.ada_weight = weight
            self.bias.data = bias
            self.weight.data = weight
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.out_act_quantizer(out)
        out = self.activation_function(out)
        if self.use_act_quant and not self.train:
            out = self.out_act_quantizer(out)
        return out

    # def init_cluster(self, x: torch.Tensor, k: int):
    #     size = x.size()
    #     n_clusters = k if k < x.numel() else x.numel()
    #     estimator = KMeans(n_clusters=n_clusters, init='k-means++')
    #     estimator.fit(x.cpu().numpy().reshape(-1,1))
    #     label_pred = estimator.labels_
    #     centroids = estimator.cluster_centers_
    #     x = x.view(-1).cpu().numpy()
    #     for i in range(np.shape(x)[0]):
    #         x[i] = centroids[label_pred[i]]
    #     x = torch.from_numpy(x).cuda()
    #     x = x.reshape(size)
    #     return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def set_save_params(self, save_params: bool = False):
        self.save_params = save_params

