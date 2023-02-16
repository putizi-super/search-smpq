import torch
import torch.nn as nn

from .base_uaq import UniformAffineQuantizer
from .base_lsq import LSQQuantizer
from .base_ristretto import RistrettoQuantizer
from .brecq_layer import QuantModule, StraightThrough

import sys
sys.path.append('../../')
from models.imagenet.resnet import BasicBlock, Bottleneck
# from models.imagenet.mobilefacenet_lite import Bottleneck_face
from models.imagenet.regnet import ResBottleneckBlock
from models.imagenet.mobilenet_v2 import InvertedResidual
# from models.imagenet.resnet_caffe import Bottleneck_caffe
# from models.imagenet.resnet_caffe import BasicBlock


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        # self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer = LSQQuantizer(**act_quant_params)
        # self.act_quantizer = RistrettoQuantizer(**act_quant_params)
        
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

    def set_save_params(self, save_params: bool = False):
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_save_params(save_params)



class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, bias_quant_params, act_quant_params)
        # self.conv1.activation_function = basic_block.relu1
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, bias_quant_params, act_quant_params)

        # modify the activation function to ReLU
        # self.activation_function = basic_block.relu2
        self.activation_function = nn.ReLU(inplace = True)

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, bias_quant_params, act_quant_params)
        # copying all attributes in original block
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += residual
        if self.use_act_quant:
            out = self.act_quantizer(out)
        out = self.activation_function(out)
        return out

# class QuantBottleneckcaffe(BaseQuantBlock):
#     def __init__(self, bottleneck: Bottleneck_caffe, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, bias_quant_params, act_quant_params)
#         self.relu1 = nn.ReLU(inplace = True)
#         self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, bias_quant_params, act_quant_params)
#         self.relu2 = nn.ReLU(inplace = True)
#         self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, bias_quant_params, act_quant_params)
#         self.activation_function = nn.ReLU(inplace = True)

#         if bottleneck.downsample is None:
#             self.downsample = None
#         else:
#             self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, bias_quant_params, act_quant_params)
#         self.stride = bottleneck.stride
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.conv3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)
        
#         out += residual
#         if self.use_act_quant:
#             out = self.act_quantizer(out)
#         out = self.activation_function(out)
#         return out


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, bias_quant_params, act_quant_params)
        # self.conv1.activation_function = bottleneck.relu1
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, bias_quant_params, act_quant_params)
        # self.conv2.activation_function = bottleneck.relu2
        self.relu2 = nn.ReLU(inplace = True)
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, bias_quant_params, act_quant_params)

        # modify the activation function to ReLU
        # self.activation_function = bottleneck.relu3
        self.activation_function = nn.ReLU(inplace = True)

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, bias_quant_params, act_quant_params)
        # copying all attributes in original block
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out += residual
        if self.use_act_quant:
            out = self.act_quantizer(out)
        out = self.activation_function(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, bias_quant_params, act_quant_params),
                nn.ReLU6(),
                QuantModule(inv_res.conv[3], weight_quant_params, bias_quant_params, act_quant_params),
            )
            # self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, bias_quant_params, act_quant_params),
                nn.ReLU6(),
                QuantModule(inv_res.conv[3], weight_quant_params, bias_quant_params, act_quant_params),
                nn.ReLU6(),
                QuantModule(inv_res.conv[6], weight_quant_params, bias_quant_params, act_quant_params),
            )
            # self.conv[0].activation_function = nn.ReLU6()
            # self.conv[1].activation_function = nn.ReLU6()

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        # out = self.activation_function(out)
        # if self.use_act_quant:
        #     out = self.act_quantizer(out)
        return out

# class QuantBottleneck_face(BaseQuantBlock):
#     def __init__(self, bottleneck: Bottleneck_face, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.connect = bottleneck.connect
#         self.conv = nn.Sequential(
#             QuantModule(bottleneck.conv[0], weight_quant_params, bias_quant_params, act_quant_params),
#             nn.ReLU(inplace = True),
#             QuantModule(bottleneck.conv[3], weight_quant_params, bias_quant_params, act_quant_params),
#             nn.ReLU(inplace = True),
#             QuantModule(bottleneck.conv[6], weight_quant_params, bias_quant_params, act_quant_params),
#         )

#     def forward(self, x):
#         if self.connect:
#             out = x + self.conv(x)
#             if self.use_act_quant:
#                 out = self.act_quantizer(out)
#         else:
#             out = self.conv(x)
#         return out

specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    # Bottleneck_face: QuantBottleneck_face,
    # Bottleneck_caffe: QuantBottleneckcaffe,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual
}

# ResBottleneckBlock: QuantResBottleneckBlock,
