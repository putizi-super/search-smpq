import torch.nn as nn
from .brecq_layer import QuantModule, StraightThrough
from .brecq_block import specials, BaseQuantBlock

import sys

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, bias_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, bias_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
        #   把各残差块对象转化为有着量化参数的残差块对象 setattr（）设置属性函数
            if (type(child_module)) in specials:
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, bias_quant_params, act_quant_params))
        #   把各层转化为QuantModel类型
            elif isinstance(child_module, (nn.Conv2d)):
                setattr(module, name, QuantModule(child_module, 2.4, weight_quant_params, bias_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
            
            elif isinstance(child_module, (nn.Linear)):
                # act_quant_params['scale_method'] = 'cross_entropy'
                setattr(module, name, QuantModule(child_module, 2.4, weight_quant_params, bias_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, bias_quant_params, act_quant_params)
    # 设置量化状态，如果QuantModule是BaseQuantBlock的话就把weight 和 activation 的量化状态设置为false
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
            # if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
    
    def set_save_params(self, save_params: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_save_params(save_params)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].in_act_quantizer.bitwidth_refactor(8)
        module_list[0].out_act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].out_act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def set_input_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].use_in_quant = True
