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

class RistrettoLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_bit = 8, a_bit = 8, pruning_rate=0.5, num = 1024, QInput = True, bSetQ = True):
        super(RistrettoLinear, self).__init__(in_features, out_features, bias=bias)
        self.quan_input = QInput
        self.is_quan = bSetQ        
        self.w_bit = w_bit
        self.a_bit = a_bit 
        self.pruning_rate = pruning_rate
        self.num = num
        self.neg_thd = - 2 ** (self.w_bit - 1) 
        self.pos_thd =  2 ** (self.w_bit - 1) - 1
 
        self.w_alpha = Parameter(torch.tensor(1.0,dtype=torch.float32)) # per-tensor
        self.b_alpha = Parameter(torch.tensor(1.0,dtype=torch.float32)) # per-tensor
        self.a_alpha = Parameter(torch.tensor(1.0,dtype=torch.float32)) # per-tensor

    def ristretto_weights_function(self,x):
        if self.w_bit == 32:
            mask = pruning(torch.abs(x),self.pruning_rate)
            x = x * mask
            weight_q = x
        else:
            mask = pruning(torch.abs(x),self.pruning_rate)
            x = x * mask
            scale = (2.0 ** round_pass(torch.log2(self.w_alpha)) / (2 ** (self.w_bit - 1) - 1))
            x = x / scale
            x = torch.clamp(x, self.neg_thd, self.pos_thd)
            x = round_pass(x)
            weight_q = x * scale

        return weight_q

    def ristretto_bias_function(self,x):
        if self.w_bit == 32:
            bias_q = x
        else:
            scale = (2.0 ** round_pass(torch.log2(self.b_alpha)) / (2 ** (self.w_bit - 1)))
            x = x / scale
            x = torch.clamp(x, self.neg_thd, self.pos_thd)
            x = round_pass(x)
            bias_q = x * scale 

        return bias_q

    def ristretto_activation_function(self,x):
        if self.a_bit == 32:
            activation_q = x
        else:
            scale = (2.0 ** round_pass(torch.log2(self.a_alpha)) / ((2.0 ** (self.a_bit - 1.0) -1)))
            x = x / scale
            x = torch.clamp(x, self.neg_thd, self.pos_thd)
            x = round_pass(x)
            activation_q = x * scale
        return activation_q


    def forward(self, x):
        if self.is_quan:
            # Weight 
            Qweight = self.ristretto_weights_function(self.weight)
            self.weight.data = Qweight

            Qbias = self.bias		
            if self.bias is not None:
                Qbias = self.ristretto_bias_function(self.bias)
            self.bias.data = Qbias

            # Input(Activation)
            Qactivation = x
            if self.quan_input:       
                Qactivation = self.ristretto_activation_function(x)
                                            
            output = F.linear(Qactivation, Qweight, Qbias)
            
        else:
            output =  F.linear(x, self.weight, self.bias)

        return output