import warnings
import torch
import torch.nn as nn

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


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        if self.channel_wise == False:
            delta_grad_scale = 1.0 / ((self.n_levels * x.numel()) ** 0.5)
        else:
            delta_grad_scale = 1.0 / ((self.n_levels * x.numel()/x.size(1)) ** 0.5)

        delta = grad_scale(self.delta, delta_grad_scale)
        x_int = round_ste(x / delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
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

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

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
                        delta = (2 * new_max) / (2 ** self.n_bits - 1)
                        zero_point = (new_max / delta).round() if x_min < 0 else 0
                        # re-calculate the scale delta if zero-point is not 0,
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max):
        delta = (2 * max) / (2 ** self.n_bits - 1)
        # we assume weight quantization is always signed
        zero_point = (max / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},leaf_param={leaf_param}'
        return s.format(**self.__dict__)