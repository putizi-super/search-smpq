import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def grad_scale(x, scale):
    y = x.abs()
    y_grad = x.abs() * scale
    return (y - y_grad).detach() + y_grad

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    # return (x.round() - x).detach() + x
    return ((x+0.5).floor() - x).detach() + x

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

def cosine_similarity_loss(output, target, eps=0.0000001):

    output_net = output.view(output.size(0), -1)
    target_net = target.view(target.size(0), -1)
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1,
                                                    keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1,
                                                      keepdim=True)

    # Calculate the KL-divergence
    loss = torch.sum(target_similarity * torch.log(
        (target_similarity + eps) / (model_similarity + eps)))

    return loss

class RistrettoQuantizer(nn.Module):
    """
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, p: float = 2.4, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max', 
                 leaf_param: bool = False):
        super(RistrettoQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.pos_thd =  2 ** (self.n_bits - 1) - 1
        self.neg_thd = - 2 ** (self.n_bits - 1)
        self.s_max = None
        self.inited = False
        self.p = p
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                s_max = self.init_quantization_scale(x, self.channel_wise, self.p)
                self.s_max = torch.nn.Parameter(s_max)
            else:
                self.s_max = self.init_quantization_scale(x, self.channel_wise, self.p)
            self.inited = True

        s_scale = (2.0 ** round_ste(torch.log2(self.s_max)) / ((2.0 ** (self.n_bits - 1.0) )))
        x = x / s_scale
        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = round_ste(x)
        x_dequant = x_quant * s_scale
        return x_dequant


    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, p: float = 2.4):
        s = None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            s = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                s[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                s = s.view(-1, 1, 1, 1)
            else:
                s = s.view(-1, 1)
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

                s = float(x_max - x_min) / 2.0
                # if s < 1e-8:
                #     warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                #     s = 1e-8
                s = torch.tensor(s).type_as(x)

            elif self.scale_method == 'mse':
                # we always use symmetric quantization in mse mode
                x_absmax = x.abs().max()
                x_min = x.min().item()
                best_score = 1000
                for i in range(500):
                    new_max = x_absmax * (1.0 - (i * 0.002))
                    x_q = self.quantize(x, new_max)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q,  2.4, reduction='all')
                    # score = cosine_similarity_loss(x, x_q)
                    if score < best_score:
                        best_score = score
                        # s = (2.0 ** round_ste(torch.log2(new_max)) / ((2.0 ** (self.n_bits - 1.0) )))
                        # s = (2 * new_max) / (2 ** self.n_bits - 1)
                        s = new_max
                        # re-calculate the scale delta if zero-point is not 0,
            
            elif self.scale_method == 'cross_entropy':
                # we always use symmetric quantization in cross_entropy mode
                x_absmax = x.abs().max()
                x_min = x.min().item()
                best_score = 1000
                for i in range(500):
                    new_max = x_absmax * (1.0 - (i * 0.002))
                    x_q = self.quantize(x, new_max)
                    score = 0.01 * cross_entropy(F.softmax(x_q,dim=1), F.softmax(x,dim=1)) + lp_loss(x, x_q,  2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        s = new_max
            else:
                raise NotImplementedError

        return s

    def quantize(self, x, max):
        # s = (2 * max) / (2 ** self.n_bits - 1)
        s = (2.0 ** round_ste(torch.log2(max)) / ((2.0 ** (self.n_bits - 1.0) )))
        # we assume weight quantization is always signed
        beta = 0.0
        x = x / s
        x = torch.clamp(x, self.neg_thd, self.pos_thd)
        x_quant = torch.floor(x+0.5)
        # x_quant = torch.round(x)
        x_float_q = x_quant * s
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