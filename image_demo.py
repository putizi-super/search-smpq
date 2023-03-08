# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import time
from importlib import import_module

import sys
sys.path.append('..')
from data import imagenet
from utils.options import parser
import utils.common as utils
from tricks.smoothing import LabelSmoothing
from tricks.mixup import NLLMultiLabelSmooth, MixUpWrapper
import models.imagenet as models

from modules.brecq.recon_layer import layer_reconstruction
from modules.brecq.recon_block import block_reconstruction
from modules.brecq.brecq_layer import QuantModule
from modules.brecq.brecq_block import BaseQuantBlock
from modules.brecq.brecq_model import QuantModel
from tool.merge_bn import search_merge_and_remove_bn



from argparse import ArgumentParser

import mmcv

from mmcls.apis import inference_model, init_model, show_result_pyplot

# quantization parameters
parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
parser.add_argument('--pruning', action='store_true', help='apply pruning weights')
parser.add_argument('--disable_8bit_head_stem', action='store_true')
parser.add_argument('--test_before_calibration', action='store_true')

parser.add_argument('--img', help='Image file')
parser.add_argument('--config', help='Config file')
parser.add_argument('--checkpoint', help='Checkpoint file')
parser.add_argument(
    '--show',
    action='store_true',
    help='Whether to show the predict results by matplotlib.')
parser.add_argument(
    '--device', default='cuda:0', help='Device used for inference')


# weight calibration parameters
parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
### 注意 the size of calibration must be 20的倍数
parser.add_argument('--num_samples', default=40, type=int, help='size of the calibration dataset')
# parser.add_argument('--num_samples', default=1000, type=int, help='size of the calibration dataset')
parser.add_argument('--weight', default=0.01, type=int, help='weight of rounding cost vs the reconstruction loss.')
parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
parser.add_argument('--b_start', default=26, type=int, help='temperature at the beginning of calibration')
parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

args = parser.parse_args()

# def quant_module_refactor( module:nn.Module):
#     for name, child_module in module.named_children():
#         if 


checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss
if args.mixup > 0.0:
    loss_func = lambda: NLLMultiLabelSmooth(args.label_smoothing)
elif args.label_smoothing > 0.0:
    loss_func = lambda: LabelSmoothing(args.label_smoothing)
loss_func = loss_func()



device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
loader = imagenet.Data(args)
trainLoader = loader.trainLoader
testLoader = loader.testLoader


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples] 


def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )
    return accuracy.avg, top5_accuracy.avg

def main():


    # build the model from a config file and a checkpoint file
    model_baseline = init_model(args.config, args.checkpoint, device=args.device)
    model_baseline1 = model_baseline.backbone

    wq_params = {'n_bits': args.w_bit, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'leaf_param': True}
    bq_params = {'n_bits': args.w_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    aq_params = {'n_bits': args.a_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}


    qnn = QuantModel(model=model_baseline1, weight_quant_params=wq_params, bias_quant_params=bq_params, act_quant_params=aq_params)
    qnn.to(device)
    qnn.eval()

        

    # test a single image
    result = inference_model(model_baseline, args.img)
    print(str(result))
    # show the results
    # print(mmcv.dump(result, file_format='json', indent=4))
    # if args.show:
    #     show_result_pyplot(model_baseline, args.img, result)


    cali_data = get_train_samples(trainLoader, num_samples=args.num_samples)
    qnn.set_quant_state(True, False)
    ### 通过一部分校准数据集进行初始化，前向传播。目的：得到一个初始化好的量化步长即可学习的近似值的参数
    _ = qnn(cali_data[:32].to(device))

    if args.test_before_calibration:
        qnn_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        print('Quantized accuracy before brecq: {}'.format(qnn_acc))


    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, weight_quant=True, opt_mode='mse')


    ### 将QuantModule中的权重给params params储存 weight 和 bias信息
    def get_params(model: nn.Module, params: list = []):
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                params.append(module.ada_weight)
                if module.bias is not None:
                    params.append(module.bias.data)
            else:
                get_params(module, params)
        return params

    def load_params(model: nn.Module, params: list):
        cnt = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = params[cnt]
                cnt += 1
                if m.bias is not None:
                    m.bias.data = params[cnt]
                    cnt += 1
            elif isinstance(m, nn.Linear):
                m.weight.data = params[cnt]
                cnt += 1
                if m.bias is not None:
                    m.bias.data = params[cnt]
                    cnt += 1
            else:
                continue
        return model
    def get_fl(model: nn.Module, weight: list = [], bias: list = [], input_fl: list = [], output_fl: list = [], eltwise: list = []):
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                weight.append(7 - torch.round(torch.log2(module.weight_quantizer.s_max)).item())
                if module.in_act_quantizer.s_max is not None:
                    input_fl.append(7 - torch.round(torch.log2(module.in_act_quantizer.s_max)).item())
                if module.out_act_quantizer.s_max is not None:
                    output_fl.append(7 - torch.round(torch.log2(module.out_act_quantizer.s_max)).item())
                if module.bias is not None:
                    bias.append(7 - torch.round(torch.log2(module.bias_quantizer.s_max)).item())
            elif isinstance(module, BaseQuantBlock):
                get_fl(module, weight, bias, input_fl, output_fl)
                if module.act_quantizer.s_max is not None:
                    output_fl.append(7 - torch.round(torch.log2(module.act_quantizer.s_max)).item())
                    eltwise.append(7 - torch.round(torch.log2(module.act_quantizer.s_max)).item())
            else:
                get_fl(module, weight, bias, input_fl, output_fl)
        return weight, bias, input_fl, output_fl, eltwise

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        global count
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    if args.pruning:
                        if count < len(pruning_rate):
                            print(pruning_rate[count])
                            layer_reconstruction(qnn, module, pruning_rate[count], **kwargs)
                            count += 1
                        else: # quant activation
                            layer_reconstruction(qnn, module, 1, **kwargs)
                    else:
                        layer_reconstruction(qnn, module, 1, **kwargs)
                    # print('Reconstruction for layer {}'.format(name))
                   
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    start = count # count the number of modules in the block
                    for name, _module in module.named_modules():
                        if isinstance(_module, QuantModule):
                            count += 1
                    print('Reconstruction for block {}'.format(name))
                
                    if args.pruning:
                        if count < len(pruning_rate): 
                            print(pruning_rate[start:count])
                            block_reconstruction(qnn, module, pruning_rate[start:count], **kwargs)
                        else: # quant activation
                            block_reconstruction(qnn, module, [1]*(count-start), **kwargs)
                    else:
                            block_reconstruction(qnn, module, [1]*(count-start), **kwargs)
            else:
                recon_model(module)

    # Start calibration 开始进行校准  因为只有权重为true,所以只对权重进行校准
    print("==>start calibration...")
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)


    # qnn_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    # print('Weight quantization accuracy: {}'.format(qnn_acc))

    if args.act_quant:
        # quantize the input of the first layer
        qnn.set_input_quantization()
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        ### 对量化步长进行初始化
        with torch.no_grad():
            _ = qnn(cali_data[:32].to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        # qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, warmup=args.warmup, act_quant=True, weight_quant=True, opt_mode='mse', lr=args.lsq_lr)
        
        # test before recon
        qnn_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

        # top1_acc, top5_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

        print('Full quantization (W{}A{}) accuracy: {}'.format(args.w_bit, args.a_bit, top1_acc))
    

    weight, bias, input_fl, output_fl, eltwise = get_fl(qnn)
    print('weight_fl: {}\nbias_fl: {}\ninput_fl: {}\noutput_fl:{}\neltwise_fl:{}'.format(weight, bias, input_fl, output_fl, eltwise))
    a_fl = input_fl + output_fl
    print(a_fl)


    # save weight
    qnn.set_save_params(save_params=True)
    with torch.no_grad():
        _ = qnn(cali_data[:32].to(device))

    # top1_acc, top5_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    params = get_params(qnn)

    # t = str(params)
    # with open('params.txt','w') as f:
    #     f.write(t)
    

    # print('==> Building mergebn model..')
    # if 'RepVGG' in args.arch:
    #     repvgg_build_func = models.get_RepVGG_func_by_name(args.arch)
    #     model_mergebn = repvgg_build_func(deploy= 'deploy').to(device)
    # else:
    #     # model_baseline = models.__dict__[args.arch](pretrained=True).to(device)
    #     model_mergebn = models.__dict__[args.arch]().to(device)
    # search_merge_and_remove_bn(model_mergebn)


    # load params   
    model_mergebn = load_params(model_baseline1, params)
    
    # top1_acc, top5_acc = test(model_mergebn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    
    

    model_state_dict = model_mergebn.state_dict()

    state = {
        'state_dict': model_state_dict,
        'weight_fl': weight,
        'bias_fl': bias,
        'a_fl' : a_fl,
    }
    checkpoint.save_model(state, is_best=True)
    # logger.info('Quantization Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3}'.format(float(top1_acc), float(top5_acc)))
    # test a single image
    model_baseline.backbone = model_mergebn
    result = inference_model(model_baseline, args.img)
    print(str(result))

if __name__ == '__main__':
    main()
    