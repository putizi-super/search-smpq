import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

def grad_scale(x, scale):
    y = x.abs()
    y_grad = x.abs() * scale
    return (y - y_grad).detach() + y_grad

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def cross_entropy(pred, tgt):
    return (-tgt*torch.log(pred)).mean()

class LSQQuantizer(nn.Module):
    """
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(LSQQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.pos_thd =  2 ** (self.n_bits - 1) - 1
        self.neg_thd = - 2 ** (self.n_bits - 1)
        self.s = None
        self.beta = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                s, self.beta = self.init_quantization_scale(x, self.channel_wise)
                self.s = torch.nn.Parameter(s)
            else:
                self.s, self.beta = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        if self.channel_wise == False:
            s_grad_scale = 1.0 / ((self.pos_thd * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.pos_thd * x.numel()/x.size(1)) ** 0.5)

        s_scale = grad_scale(self.s, s_grad_scale)
        x = (x - self.beta) / s_scale
        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = round_ste(x)
        x_dequant = x_quant * self.s + self.beta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        s, beta = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            s = x_max.clone()
            beta = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                s[c], beta[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                s = s.view(-1, 1, 1, 1)
                beta = beta.view(-1, 1, 1, 1)
            else:
                s = s.view(-1, 1)
                beta = beta.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                s = float(x_max - x_min) / (2 ** self.n_bits - 1)
                if s < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    s = 1e-8

                beta = x_min + float(x_max - x_min) / 2.0
                s = torch.tensor(s).type_as(x)

            elif self.scale_method == 'mse':
                # we always use symmetric quantization in mse mode
                x_absmax = x.abs().max()
                x_min = x.min().item()
                best_score = 1000
                for i in range(80):
                    new_max = x_absmax * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        s = (2 * new_max) / (2 ** self.n_bits - 1)
                        beta = 0.0
                        # re-calculate the scale delta if zero-point is not 0,
            
            elif self.scale_method == 'cross_entropy':
                # we always use symmetric quantization in cross_entropy mode
                x_absmax = x.abs().max()
                x_min = x.min().item()
                best_score = 1000
                for i in range(500):
                    print(best_score)
                    new_max = x_absmax * (1.0 - (i * 0.002))
                    x_q = self.quantize(x, new_max)
                    score = 0.1 * cross_entropy(F.softmax(x_q,dim=1), F.softmax(x,dim=1)) + lp_loss(x, x_q,  2.4, reduction='all')
                    print(best_score)
                    if score < best_score:
                        best_score = score
                        s = (2 * new_max) / (2 ** self.n_bits - 1)
                        beta = 0.0   

            else:
                raise NotImplementedError

        return s, beta

    def quantize(self, x, max):
        s = (2 * max) / (2 ** self.n_bits - 1)
        # we assume weight quantization is always signed
        beta = 0.0
        x = (x - beta) / s
        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = torch.round(x)
        x_float_q = x_quant * s + beta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.pos_thd =  2 ** (self.n_bits - 1) - 1
        self.neg_thd = - 2 ** (self.n_bits - 1)

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)