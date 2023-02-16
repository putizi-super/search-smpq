import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

class PACT(Module):
    def __init__(self):
        super(PACT, self).__init__()
        self.alpha = Parameter(torch.tensor(10.0,dtype=torch.float32))
    def forward(self, input):
        output =  0.5*(torch.abs(input) - torch.abs(input-self.alpha) + self.alpha)
        return output

class Scale_PACT(Module):
    def __init__(self):
        super(Scale_PACT, self).__init__()
        self.k = Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.b = Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.alpha = Parameter(torch.tensor(6.0,dtype=torch.float32))
    def forward(self, input):
        input = self.k*(input+self.b)
        output =  0.5*(torch.abs(input) - torch.abs(input-self.alpha) + self.alpha)
        return output

class Scale_HardTanh(Module):
    def __init__(self,out_chn,bit,per_channel=True):
        super(Scale_HardTanh, self).__init__()
        self.bit = bit
        if per_channel == True:
            self.b1 = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
            self.b2 = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
            self.k1 = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True)
            self.k2 = nn.Parameter(0.1*torch.ones(1,out_chn,1,1), requires_grad=True)
            # if self.bit > 4:
            #     self.k2 = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
            # elif self.bit == 4 or bit == 3:
            #     self.k2 = nn.Parameter(0.01*torch.ones(1,out_chn,1,1), requires_grad=True)
            # elif self.bit == 2:
            #     self.k2 = nn.Parameter(0.1*torch.ones(1,out_chn,1,1), requires_grad=True)
            # elif self.bit == 1:
            #     self.k2 = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True) 
        else:
            self.bit = bit
            self.b1 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.b2 = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.k1 = nn.Parameter(torch.ones(1), requires_grad=True)
            self.k2 = nn.Parameter(0.1*torch.ones(1), requires_grad=True)

            # if self.bit > 4:
            #     self.k2 = nn.Parameter(torch.zeros(1), requires_grad=True)
            # elif self.bit == 4 or bit == 3:
            #     self.k2 = nn.Parameter(0.01*torch.ones(1), requires_grad=True)
            # elif self.bit == 2:
            #     self.k2 = nn.Parameter(0.1*torch.ones(1), requires_grad=True)
            # elif self.bit == 1:
            #     self.k2 = nn.Parameter(torch.ones(1), requires_grad=True)   

    def forward(self, input):
        # if self.bit == 1:
        #     input =  input+self.b1
        #     mask = input > 0
        #     output = input*mask.type_as(input)*self.k1 + input*(1.0-mask.type_as(input))*self.k2 + self.b2
        # elif self.bit <= 3:
        #     mask = input > 0
        #     output = input*mask.type_as(input) + input*(1.0-mask.type_as(input))*self.k2
        # else:
        #     mask = input > 0
        #     output = input*mask.type_as(input)

        input =  input+self.b1
        mask = input > 0
        output = input*mask.type_as(input)*self.k1 + input*(1.0-mask.type_as(input))*self.k2 + self.b2
        # output = torch.max(input*self.k1+self.b1,input*self.k2+self.b2)
        return output


class PACT_PLUS(Module):
    def __init__(self):
        super(PACT_PLUS, self).__init__()
        self.k = Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.alpha = Parameter(torch.tensor(5.0,dtype=torch.float32))
        self.beta = Parameter(torch.tensor(-5.0,dtype=torch.float32))
        # self.beta = 0.0
    def forward(self, input):
        output =  0.5*(torch.abs(self.k * input-self.beta) - torch.abs(self.k * input-self.alpha) + self.alpha + self.beta)
        # output =  0.5*(torch.abs(input-self.beta) - torch.abs(input-self.alpha) + self.alpha + self.beta)
        return output

 
class Maxout(Module):
    def __init__(self,level):
        super(Maxout, self).__init__()
        self.k = Parameter(torch.ones(level,1,1,1,1))
        self.b = Parameter(torch.zeros(level,1,1,1,1))
    def forward(self, input):
        output = input.unsqueeze(0)*self.k +self.b
        output = torch.max(output,dim=0)[0]
        # print(output)
        return output

class PRelu(Module):
    def __init__(self):
        super(PRelu, self).__init__()
        self.k = Parameter(torch.tensor(0.1,dtype=torch.float32))
    def forward(self, input):
        mask = input > 0
        output = input*mask.float() + input*(1.0-mask.float())*self.k
        return output

class Scale_PRelu(Module):
    def __init__(self):
        super(Scale_PRelu, self).__init__()
        self.b1 = Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.b2 = Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.k1 = Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.k2 = Parameter(torch.tensor(0.1,dtype=torch.float32))
    def forward(self, input):
        input =  input+self.b1
        mask = input > 0
        output = input*mask.float()*self.k1 + input*(1.0-mask.float())*self.k2 + self.b2
        return output