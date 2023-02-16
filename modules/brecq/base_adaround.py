import torch
import math
from torch import nn
from typing import Union
from .base_uaq import UniformAffineQuantizer, round_ste
from .base_lsq import LSQQuantizer
from .base_ristretto import RistrettoQuantizer

def pruning(x,rate):
    k_num = int((1.0-rate) * len(x.view(-1)))
    if k_num >= len(x.view(-1)):
        k_num = len(x.view(-1))-1
    elif k_num < 0:
        k_num = 0
    input_s = torch.sort(x.view(-1))
    val = input_s[0][k_num] 
    mask = x >= val
    return mask.float()

def channel_keep(x,num):
    x1, _ = torch.sort(x.view(x.size(0), -1), 1, descending = True)
    if x1.size(1) <= num:
        mask = torch.ones_like(x)
    else:
        val = x1[:, num]
        val = val.reshape(x.size(0), 1, 1, 1)
        mask = x > val
    return mask.float()

class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, quantizer: Union[UniformAffineQuantizer, LSQQuantizer, RistrettoQuantizer], weight_tensor: torch.Tensor, pruning_rate= 1, num = 1024, round_mode='learned_hard_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = quantizer.n_bits
        self.sym = quantizer.sym
        self.pruning_rate = pruning_rate
        self.num = num
        if isinstance(quantizer, UniformAffineQuantizer):
            self.quantizer = 'uaq'
            self.delta = quantizer.delta
            self.zero_point = quantizer.zero_point
            self.n_levels = quantizer.n_levels
        elif isinstance(quantizer, LSQQuantizer):
            self.quantizer = 'lsq'
            self.s = quantizer.s
            self.beta = quantizer.beta
            self.pos_thd =  quantizer.pos_thd
            self.neg_thd = quantizer.neg_thd
        elif isinstance(quantizer, RistrettoQuantizer):
            self.quantizer = 'ristretto'
            self.s_max = quantizer.s_max
            self.pos_thd =  quantizer.pos_thd
            self.neg_thd = quantizer.neg_thd
        else:
            raise NotImplementedError

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        # self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.quantizer == 'uaq':
            if self.round_mode == 'nearest':
                x_int = torch.round(x / self.delta)
            elif self.round_mode == 'nearest_ste':
                x_int = round_ste(x / self.delta)
            elif self.round_mode == 'stochastic':
                x_floor = torch.floor(x / self.delta)
                rest = (x / self.delta) - x_floor  # rest of rounding
                x_int = x_floor + torch.bernoulli(rest)
                print('Draw stochastic sample')
            elif self.round_mode == 'learned_hard_sigmoid':
                x_floor = torch.floor(x / self.delta)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            else:
                raise ValueError('Wrong rounding mode')

            x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * self.delta


        elif self.quantizer == 'lsq':
            if self.round_mode == 'nearest':
                x_int = torch.round((x - self.beta) / self.s)
            elif self.round_mode == 'nearest_ste':
                x_int = round_ste((x - self.beta) / self.s)
            elif self.round_mode == 'stochastic':
                x_floor = torch.floor((x - self.beta) / self.s)
                rest = ((x - self.beta) / self.s) - x_floor  # rest of rounding
                x_int = x_floor + torch.bernoulli(rest)
                print('Draw stochastic sample')
            elif self.round_mode == 'learned_hard_sigmoid':
                x_floor = torch.floor((x - self.beta) / self.s)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            else:
                raise ValueError('Wrong rounding mode')

            x_quant = torch.clamp(x_int, self.neg_thd, self.pos_thd)
            x_float_q = x_quant * self.s + self.beta      

        elif self.quantizer == 'ristretto':
            scale = (2.0 ** round_ste(torch.log2(self.s_max)) / ((2.0 ** (self.n_bits - 1.0) )))
            if self.round_mode == 'nearest':
                x_int = torch.round(x / scale)
            elif self.round_mode == 'nearest_ste':
                x_int = round_ste(x / scale)
            elif self.round_mode == 'stochastic':
                x_floor = torch.floor(x / scale)
                rest = (x / scale) - x_floor  # rest of rounding
                x_int = x_floor + torch.bernoulli(rest)
                print('Draw stochastic sample')
            elif self.round_mode == 'learned_hard_sigmoid':
                x_floor = torch.floor(x / scale)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            elif self.round_mode == 'learned_hard_tanh':
                x_floor = torch.floor(x / scale)
                if self.soft_targets:
                    x_int = x_floor + self.get_tanh_soft_targets()
                else:
                    x_int = x_floor + (self.alpha > (-0.5 * math.log((self.zeta - 0.5) / (0.5 + self.zeta)))).float() - (self.alpha < (0.5 * math.log((self.zeta - 0.5) / (0.5 + self.zeta)))).float()

            else:
                raise ValueError('Wrong rounding mode')

            x_quant = torch.clamp(x_int, self.neg_thd, self.pos_thd)
            x_float_q = x_quant * scale
            mask = pruning(torch.abs(x_float_q), self.pruning_rate)
            assert 0 == ((x_float_q != (x_float_q * mask)).sum())
            x_float_q = x_float_q * mask
            # mask = channel_keep(torch.abs(x_float_q), self.num)
            # x_float_q = x_float_q * mask
        
        else:
            raise NotImplementedError
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def get_tanh_soft_targets(self):
        return torch.clamp(torch.tanh(self.alpha) * self.zeta, -1, 1)

    def init_alpha(self, x: torch.Tensor):
        if self.quantizer == 'lsq':
            x_floor = torch.floor((x - self.beta) / self.s)
            if self.round_mode == 'learned_hard_sigmoid':
                print('Init alpha to be FP32')
                rest = ((x - self.beta) / self.s) - x_floor  # rest of rounding [0, 1)
                alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
                self.alpha = nn.Parameter(alpha)
            else:
                raise NotImplementedError
        elif self.quantizer == 'ristretto':
            scale = (2.0 ** round_ste(torch.log2(self.s_max)) / ((2.0 ** (self.n_bits - 1.0) )))
            x_floor = torch.floor(x / scale)
            if self.round_mode == 'learned_hard_sigmoid':
                print('Init alpha to be FP32')
                rest = (x / scale) - x_floor  # rest of rounding [0, 1)
                alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
                self.alpha = nn.Parameter(alpha)
            elif self.round_mode == 'learned_hard_tanh':
                rest = (x / scale) - x_floor # rest of rounding [-1,1]
                alpha = -0.5 * torch.log((self.zeta - rest) / (rest + self.zeta))
                self.alpha = nn.Parameter(alpha)
            else:
                raise NotImplementedError

        elif self.quantizer == 'uaq':
            x_floor = torch.floor(x / self.delta)
            if self.round_mode == 'learned_hard_sigmoid':
                print('Init alpha to be FP32')
                rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
                alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
                self.alpha = nn.Parameter(alpha)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
