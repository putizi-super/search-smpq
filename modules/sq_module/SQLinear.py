import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

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

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad
    
class SQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_bit = 8, a_bit = 8, pruning_rate=0.5, keep_channel=1024, QInput = True, bSetQ = True):
        super(SQLinear, self).__init__(in_features, out_features, bias=bias)
        self.quan_input = QInput
        self.is_quan = bSetQ        
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.pruning_rate = pruning_rate
        self.keep_channel = keep_channel
        self.w_bit = w_bit
        self.a_bit = a_bit 
        self.w_pos_thd =  2 ** (self.w_bit - 1) - 1
        self.w_neg_thd = - 2 ** (self.w_bit - 1)
        self.a_pos_thd = 2 ** (self.a_bit - 1) - 1 
        self.a_neg_thd = - 2 ** (self.a_bit - 1)

        self.w_s = Parameter(torch.max(torch.abs(self.weight.detach().mean() - 3.0*self.weight.detach().var()), torch.abs(self.weight.detach().mean() + 3.0*self.weight.detach().var())) / self.w_pos_thd ) # per-tensor
        self.b_s = Parameter(torch.max(torch.abs(self.weight.detach().mean() - 3.0*self.weight.detach().var()), torch.abs(self.weight.detach().mean() + 3.0*self.weight.detach().var())) / self.w_pos_thd ) # per-tensor
        # if self.a_bit > 5:
        #     self.a_s = Parameter(torch.tensor(0.1,dtype=torch.float32)) 
        # else:
        #     self.a_s = Parameter(torch.tensor(0.5,dtype=torch.float32))
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
                scaling_factor = torch.mean(abs(x),dim=1,keepdim=True)
                scaling_factor = scaling_factor.detach()
                # scaling_factor = torch.mean(abs(x)).detach()
            else:
                mask = pruning(torch.abs(x),self.pruning_rate)
                x = x * mask
                mask = channel_keep(torch.abs(x),self.keep_channel)
                x = x * mask
                sum_filter = torch.sum(abs(x),dim=1,keepdim=True)   # per-channel
                sum_mask =  torch.sum(mask,dim=1,keepdim=True)     # per-channel
                scaling_factor = (sum_filter/(sum_mask+1e-12)) 
                scaling_factor = scaling_factor.detach()
                # scaling_factor = torch.sum(abs(x))/torch.sum(mask).detach()
                
            binary_weights_no_grad = scaling_factor * torch.sign(x)
            cliped_weights = torch.clamp(x, -1.5, 1.5)
            weight_q = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            if self.pruning_rate == 1.0:
                s_grad_scale = 1.0 / ((self.w_pos_thd * x.numel()) ** 0.5)
            else:
                mask = pruning(torch.abs(x),self.pruning_rate)
                x = x * mask
                mask = channel_keep(torch.abs(x),self.keep_channel)
                x = x * mask
                s_grad_scale = 1.0 / ((self.w_pos_thd * mask.sum()) ** 0.5)
                
            s_scale = grad_scale(self.w_s, s_grad_scale)

            # s_scale =  self.w_s
            # s_scale =  self.w_s.abs()

            x = x / s_scale
            x = torch.clamp(x, self.w_neg_thd, self.w_pos_thd)
            x = round_pass(x)
            weight_q = x * s_scale

        return weight_q

    def sq_bias_function(self,x):
        if self.w_bit == 32 or self.w_bit == 1:
            bias_q = x
        else:
            scale = self.b_s
            x = x / scale
            x = torch.clamp(x, self.w_neg_thd, self.w_pos_thd)
            x = round_pass(x)
            bias_q = x * scale

        return bias_q

    def sq_activation_function(self,x):
        if self.a_bit == 32:
            activation_q = x
        elif self.a_bit == 1:
            out_forward = torch.sign(x)
            out_clip = x
            activation_q = out_forward.detach() - out_clip.detach() + out_clip
        else:
            s_grad_scale = 1.0 / ((self.a_pos_thd * x.numel()) ** 0.5)
            s_scale = grad_scale(self.a_s, s_grad_scale)
            # s_scale = self.a_s
            x = x  / s_scale
            # x = (x - self.a_beta) / s_scale
            x = torch.clamp(x, self.a_neg_thd, self.a_pos_thd)
            x = round_pass(x)
            activation_q = x * s_scale
            # activation_q = x * s_scale + self.a_beta
        return activation_q

    def forward(self, x):
        if self.is_quan:
            # Weight 
            Qweight = self.sq_weights_function(self.weight)
            Qbias = self.bias
            # Bias			
            # if self.bias is not None:
            #     Qbias = self.sq_bias_function(self.bias)

            # Input(Activation)
            Qactivation = x
            if self.quan_input:       
                Qactivation = self.sq_activation_function(x)
                
            output = F.linear(Qactivation, Qweight, Qbias)
            
        else:
            output =  F.linear(x, self.weight, self.bias)

        return output