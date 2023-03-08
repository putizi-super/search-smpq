import torch
from torch import nn

from .base_uaq import UniformAffineQuantizer,lp_loss
from .base_lsq import LSQQuantizer
from .base_ristretto import RistrettoQuantizer
from .brecq_layer import QuantModule,StraightThrough
from .brecq_model import QuantModel
from .base_adaround import AdaRoundQuantizer
from .data_utils import save_grad_data, save_inp_oup_data

def layer_reconstruction(model: QuantModel, layer: QuantModule, pruning_rate: float, cali_data: torch.Tensor,
                         batch_size: int = 20, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (15, 2),
                         warmup: float = 0.0, lr: float = 4e-5, act_quant: bool = False, weight_quant:bool = False, p: float = 2.0):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
        :param p: L_p norm minimization
    """
    # if not act_quant:
    device = next(model.parameters()).device        ## next(self.parameters()) 用于返回第一个参数  整句代码就是指定 使用和参数相同的设备
    model.set_quant_state(False, False)
    layer.set_quant_state(weight_quant, act_quant)
    round_mode = 'learned_hard_sigmoid'
    # round_mode = 'learned_hard_tanh'

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        layer.weight_quantizer = AdaRoundQuantizer(quantizer=layer.weight_quantizer, pruning_rate = pruning_rate, round_mode=round_mode, 
                                                   weight_tensor=layer.org_weight.data)
        layer.weight_quantizer.soft_targets = True

        # Set up optimizer
        print(layer.weight_quantizer.s_max)
        opt_params = [layer.weight_quantizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = []
        # Use UniformAffineQuantizer to learn delta
        if isinstance(layer.in_act_quantizer, UniformAffineQuantizer):
            if layer.in_act_quantizer.delta is not None:
                opt_params += [layer.in_act_quantizer.delta]
        elif isinstance(layer.in_act_quantizer, LSQQuantizer):
            if layer.in_act_quantizer.s is not None:
                opt_params += [layer.in_act_quantizer.s]
        elif isinstance(layer.in_act_quantizer, RistrettoQuantizer):
            if layer.in_act_quantizer.s_max is not None:
                opt_params += [layer.in_act_quantizer.s_max]
        else:
            raise NotImplementedError
        if isinstance(layer.out_act_quantizer, UniformAffineQuantizer):
            if layer.out_act_quantizer.delta is not None:
                opt_params += [layer.out_act_quantizer.delta]
        elif isinstance(layer.out_act_quantizer, LSQQuantizer):
            if layer.out_act_quantizer.s is not None:
                opt_params += [layer.out_act_quantizer.s]
        elif isinstance(layer.out_act_quantizer, RistrettoQuantizer):
            if layer.out_act_quantizer.s_max is not None:
                opt_params += [layer.out_act_quantizer.s_max]
        else:
            raise NotImplementedError
        print(opt_params)
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(layer, round_loss=loss_mode, round_mode=round_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)

    # Save data before optimizing the rounding
    # cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size)
    # if opt_mode != 'mse':
    #     cached_grads = save_grad_data(model, layer, cali_data, act_quant, batch_size=batch_size)
    # else:
    #     cached_grads = None

    # # device = 'cuda'
    # for i in range(iters):
    #     idx = torch.randperm(cached_inps.size(0))[:batch_size]
    #     cur_inp = cached_inps[idx]
    #     cur_out = cached_outs[idx]
    #     cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

    #     optimizer.zero_grad()
    #     out_quant = layer(cur_inp)

    #     err = loss_func(out_quant, cur_out, cur_grad)

    #     err.backward(retain_graph=True)
    #     optimizer.step()
    #     if scheduler:
    #         scheduler.step()

    # torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    layer.weight_quantizer.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        layer.activation_function = org_act_func


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 round_mode: str = 'learned_hard_sigmoid',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2):

        self.layer = layer
        self.round_loss = round_loss
        self.round_mode = round_mode
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            if self.round_mode == 'learned_hard_sigmoid':
                round_vals = self.layer.weight_quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()  
            elif self.round_mode == 'learned_hard_tanh':
                round_vals = self.layer.weight_quantizer.get_tanh_soft_targets()
                # round_loss += self.weight * (1 - (round_vals.abs()).pow(b)).sum()
                round_loss += self.weight * (1 - ((((round_vals - 0.5).abs() - 0.5).abs() - 0.5).abs() * 2).pow(b)).sum()

        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}\tp={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count, self.p))
        return total_loss

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))