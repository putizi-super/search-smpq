import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Function

def pruning(x,rate):
    k_num = int((1.0-rate) * len(x.view(-1)))
    if k_num >= len(x.view(-1)):
        k_num = len(x.view(-1))-1
    elif k_num < 0:
        k_num = 0
    input_s = torch.sort(x.view(-1))
    val = input_s[0][k_num] 
    mask = x > val
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

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class SignTwoOrders(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input_wrt_output2 = torch.zeros_like(grad_output)
        ge0_lt1_mask = input.ge(0) & input.lt(1)
        grad_input_wrt_output2 = torch.where(ge0_lt1_mask, (2 - 2 * input), grad_input_wrt_output2)
        gen1_lt0_mask = input.ge(-1) & input.lt(0)
        grad_input_wrt_output2 = torch.where(gen1_lt0_mask, (2 + 2 * input), grad_input_wrt_output2)
        grad_input = grad_input_wrt_output2 * grad_output

        return grad_input

class SQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                w_bit = 8, a_bit = 8, pruning_rate = 0.5, keep_channel = 1024, QInput = True, bSetQ = True):
        super(SQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quan_input = QInput
        self.is_quan = bSetQ
        self.pruning_rate = pruning_rate
        self.keep_channel = keep_channel
        self.w_bit = w_bit
        self.a_bit = a_bit 
        self.w_pos_thd = 2 ** (self.w_bit - 1) - 1
        # self.w_s = Parameter(self.weight.detach().abs().mean() * 2 / ((2 ** (self.w_bit - 1) - 1) ** 0.5)) # per-tensor
        self.w_s = Parameter(torch.max(torch.abs(self.weight.detach().mean() - 3.0*self.weight.detach().var()), torch.abs(self.weight.detach().mean() + 3.0*self.weight.detach().var())) / self.w_pos_thd ) # per-tensor
        self.a_s = Parameter(torch.tensor(0.5,dtype=torch.float32)) 
        self.a_beta = Parameter(torch.tensor(0.0,dtype=torch.float32)) 

    def sq_weights_function(self,x):
        if self.w_bit == 32:
            if self.pruning_rate == 1.0:
                weight_q = x
            else:
                mask = pruning(torch.abs(x),self.pruning_rate)
                x = x * mask
                mask = channel_keep(torch.abs(x),self.keep_channel)
                x = x * mask
                weight_q = x
        elif self.w_bit == 1:
            if self.pruning_rate == 1.0:
                scaling_factor = torch.mean(torch.mean(torch.mean(abs(x),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
                scaling_factor = scaling_factor.detach()
                # scaling_factor = torch.mean(abs(x)).detach()
            else:
                mask = pruning(torch.abs(x),self.pruning_rate)
                x = x * mask
                mask = channel_keep(torch.abs(x),self.keep_channel)
                x = x * mask
                weights_sum = torch.sum(torch.sum(torch.sum(abs(x),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
                mask_sum = torch.sum(torch.sum(torch.sum(mask,dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
                scaling_factor = (weights_sum / (mask_sum+1e-12))
                scaling_factor = scaling_factor.detach()
                # weights_sum = torch.sum(abs(x)).detach()
                # scaling_factor = (weights_sum / torch.sum(mask)).detach()

            binary_weights_no_grad = scaling_factor * torch.sign(x)
            cliped_weights = torch.clamp(x, -1.5, 1.5)
            weight_q = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            w_pos_thd = 2 ** (self.w_bit - 1) - 1
            w_neg_thd = - 2 ** (self.w_bit - 1)
            if self.pruning_rate == 1.0:
                s_grad_scale = 1.0 / ((w_pos_thd * x.numel()) ** 0.5)
            else:
                mask = pruning(torch.abs(x),self.pruning_rate)
                x = x * mask
                mask = channel_keep(torch.abs(x),self.keep_channel)
                x = x * mask
                s_grad_scale = 1.0 / ((w_pos_thd * mask.sum()) ** 0.5)
            s_scale = grad_scale(self.w_s, s_grad_scale)
            # s_scale = self.w_s
            x = x / s_scale
            x = torch.clamp(x, w_neg_thd, w_pos_thd)
            x = round_pass(x)
            weight_q = x * s_scale

        return weight_q

    def sq_activation_function(self,x):
        if self.a_bit == 32:
            activation_q = x
        elif self.a_bit == 1:
            # out_forward = torch.sign(x)
            # out_clip = torch.clamp(x, -1.5, 1.5)
            # activation_q = out_forward.detach() - out_clip.detach() + out_clip
            activation_q = SignTwoOrders.apply(x)

        else:
            if x.size(1) == 3:
                a_pos_thd = 2 ** (self.a_bit - 1) - 1
                a_neg_thd = - 2 ** (self.a_bit - 1)
            else:
                a_pos_thd =  2 ** self.a_bit - 1
                a_neg_thd =  0
                # a_pos_thd = 2 ** (self.a_bit - 1) - 1
                # a_neg_thd = - 2 ** (self.a_bit - 1)

            s_grad_scale = 1.0 / ((a_pos_thd * x.numel()) ** 0.5)
            s_scale = grad_scale(self.a_s, s_grad_scale)

            # s_scale = self.a_s
            x = x / s_scale 
            # x = (x - self.a_beta) / s_scale
            x = torch.clamp(x, a_neg_thd, a_pos_thd)
            x = round_pass(x)
            activation_q = x * s_scale
            # activation_q = x * s_scale + self.a_beta

        return activation_q

    def forward(self, x):
        if self.is_quan:
            # Weight
            # print(self.weight)
            # sys.exit()
            Qweight = self.sq_weights_function(self.weight)

            # Bias		
            Qbias = self.bias	
            # if self.bias is not None:
            #     Qbias = self.sq_bias_function(self.bias)

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
                Qactivation = self.sq_activation_function(x)

            # print(Qweight)

            # sys.exit()
            
            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output