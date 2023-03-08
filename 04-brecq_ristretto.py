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
# import torchvision.models as models


from modules.brecq.recon_layer import layer_reconstruction
from modules.brecq.recon_block import block_reconstruction
from modules.brecq.brecq_layer import QuantModule
from modules.brecq.brecq_block import BaseQuantBlock
from modules.brecq.brecq_model import QuantModel
from tool.merge_bn import search_merge_and_remove_bn


# quantization parameters
parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
parser.add_argument('--pruning', action='store_true', help='apply pruning weights')
parser.add_argument('--disable_8bit_head_stem', action='store_true')
parser.add_argument('--test_before_calibration', action='store_true')
parser.add_argument(
    '--baseline_model',
    type=str,
    default='./baseline/imagenet/resnet18.pth',
    # default='./baseline/imagenet/darknet19.pth',
    # default='./baseline/imagenet/regnet_y_400.pth',
    help='Path to the model wait for test. default:None'
)

# weight calibration parameters
parser.add_argument('--num_samples', default=1000, type=int, help='size of the calibration dataset')
parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
parser.add_argument('--weight', default=0.01, type=int, help='weight of rounding cost vs the reconstruction loss.')
parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
parser.add_argument('--b_start', default=26, type=int, help='temperature at the beginning of calibration')
parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
# parser.add_argument('--warmup1', default=0.2, type=float, help='in the warmup period no regularization is applied')
parser.add_argument('--step', default=20, type=int, help='record snn output per step')

# activation calibration parameters
parser.add_argument('--iters_a', default=4000, type=int, help='number of iteration for LSQ')
parser.add_argument('--lsq_lr', default=5e-6, type=float, help='learning rate for LSQ')
parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss
if args.mixup > 0.0:
    loss_func = lambda: NLLMultiLabelSmooth(args.label_smoothing)
elif args.label_smoothing > 0.0:
    loss_func = lambda: LabelSmoothing(args.label_smoothing)
loss_func = loss_func()


print('==> Preparing data..')
loader = imagenet.Data(args)
trainLoader = loader.trainLoader
testLoader = loader.testLoader

count = 0

### 这个rate怎么来的？
# resnet18_caffe channel_keep /loss 3.1693
# pruning_rate=[0.2509892381838768, 0.1949955777182775, 0.15481105457277416, 0.181482879280324, 0.15236411504078137, 0.12624166044322543, 0.19945650833123205, 0.14867201615531073, 0.09744269325603885, 0.16507781611943093, 0.22461711527148995, 0.19174842125328845, 0.13530936319651926, 0.2769501456887007, 0.1281951215877744, 0.09779858333620649, 0.13675934743598922, 0.18702198771408263, 0.2592740954241761, 0.16431281448915383, 0.15246512653703692, 0.2972427183432819]

# resnet50_caffe
# pruning_rate=[0.2551811675419373, 0.10883450978339962, 0.22583542532487397, 0.19547508745002112, 0.11369726586315448, 0.16172124328818335, 0.157331220246815, 0.20782626185407252, 0.14391872824829183, 0.16731433895536404, 0.16290334019176939, 0.07652782086899101, 0.10724719780714069, 0.22040511663401713, 0.1833692878525247, 0.2356364290238544, 0.10622556191256613, 0.2113188295263494, 0.15299678063165925, 0.12861094312257557, 0.1634710847992596, 0.12435095588018703, 0.1421670802928368, 0.09432220564206424, 0.16910182541536306, 0.13505329597556698, 0.15073397260178548, 0.2088968274958775, 0.12160138166123047, 0.11254312615419304, 0.24582578448397757, 0.10120886520052749, 0.1716853042194909, 0.1179815795030305, 0.06870041233210827, 0.21524053356981737, 0.1920003803613265, 0.19109731990415102, 0.16537330873955833, 0.2374034855171861, 0.19063310269885037, 0.09511070407087727, 0.17917524639950763, 0.1522430506069873, 0.10205502360726787, 0.1797667607105435, 0.0974781111479703, 0.15958162315516056, 0.056207185648549735, 0.1495195241948409, 0.3634150746772293, 0.31189005300252604, 0.1266773460002937, 0.23611228824609895]

# darknet19
# pruning_rate=[0.22601208873400958, 0.10288574462221352, 0.12460391727875142, 0.17003247327033716, 0.15059471403352395, 0.24713115197214544, 0.2146920926486741, 0.217769548231671, 0.14281390810891315, 0.2720822657291109, 0.1415971844663011, 0.41014952696893975, 0.14653270909677835, 0.1556220252654409, 0.32767339395800604, 0.17756908008217032, 0.36561126976494673, 0.1126105376074497, 0.1876873927501988]

# darknet53
# pruning_rate=[0.0935725427090472, 0.3154058397045244, 0.11794029697567839, 0.1432565571054809, 0.12617671504859643, 0.15633130678494384, 0.11791512934373297, 0.27530598460035133, 0.1994298906838725, 0.16312778635619446, 0.06909462371086182, 0.2222056118678103, 0.24023209541719429, 0.18507532595776838, 0.24088904158628574, 0.12973209512403377, 0.19017652023731293, 0.09952402336464297, 0.16166187385514313, 0.22247101222482346, 0.2842359207043734, 0.1793250343568802, 0.10295495957551239, 0.21705065127991424, 0.24274808901803896, 0.12712101799354758, 0.14232817152094768, 0.05532790439637863, 0.1312041474166296, 0.09365893318688293, 0.06651964030264898, 0.17697121839057567, 0.039704058786716075, 0.1005127183261051, 0.07097954702629064, 0.2032882167376095, 0.27948468200731985, 0.2455022121993691, 0.14888163989810094, 0.14654239236545236, 0.18058964137790903, 0.10910065975462932, 0.12865591856979885, 0.1580572694153708, 0.24204536549085634, 0.1478525526381133, 0.3501761085227733, 0.19928817771176344, 0.1876057514025239, 0.160909003555216, 0.2077678718854213, 0.1563643895789183, 0.37369412979282957]



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

def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples] 

def main():
    print('==> Building model..')
    if 'RepVGG' in args.arch:
        repvgg_build_func = models.get_RepVGG_func_by_name(args.arch)
        model_baseline = repvgg_build_func(deploy= 'deploy').to(device)
        print(args.baseline_model)
    else:
        # model_baseline = models.__dict__[args.arch](pretrained=True).to(device)
        model_baseline = models.__dict__[args.arch]().to(device)

    if args.baseline == True:
        ckpt = torch.load(args.baseline_model, map_location=device)
        model_baseline.load_state_dict(utils.convert_keys(model_baseline, ckpt))
        # baseline_acc = test(model_baseline, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    search_merge_and_remove_bn(model_baseline)   # ?为什么
    wq_params = {'n_bits': args.w_bit, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'leaf_param': True}
    bq_params = {'n_bits': args.w_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    aq_params = {'n_bits': args.a_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    # 创建一个 QuantModel类,且该类有了各种了量化参数 
    qnn = QuantModel(model=model_baseline, weight_quant_params=wq_params, bias_quant_params=bq_params, act_quant_params=aq_params)
    qnn.to(device)
    qnn.eval()

    # pruning need be ignored !
    # if not args.disable_8bit_head_stem:
    #     print('Setting the first and the last layer to 8-bit')
    #     qnn.set_first_last_layer_to_8bit()

    print(qnn)
    ### 获得一部分校准数据集
    cali_data = get_train_samples(trainLoader, num_samples=args.num_samples)

    # Initialize weight quantization parameters
    ### qnn的weight量化状态设置为true activation的量化状态设置为false   注：量化分为两阶段，先对权重量化，后对激活量化
    qnn.set_quant_state(True, False)
    ### 通过一部分校准数据集进行初始化，前向传播。目的：得到一个初始化好的量化步长即可学习的近似值的参数
    _ = qnn(cali_data[:32].to(device))

    ### 先测试刚初始化的模型
    if args.test_before_calibration:
        qnn_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
        print('Quantized accuracy before brecq: {}'.format(qnn_acc))

    ### 之后是初始化后的模型进行校准
    # sys.exit()
    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, weight_quant=True, opt_mode='mse')

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
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    qnn_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))
    print('Weight quantization accuracy: {}'.format(qnn_acc))

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

        top1_acc, top5_acc = test(qnn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

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
    print('==> Building mergebn model..')
    if 'RepVGG' in args.arch:
        repvgg_build_func = models.get_RepVGG_func_by_name(args.arch)
        model_mergebn = repvgg_build_func(deploy= 'deploy').to(device)
    else:
        # model_baseline = models.__dict__[args.arch](pretrained=True).to(device)
        model_mergebn = models.__dict__[args.arch]().to(device)
    search_merge_and_remove_bn(model_mergebn)

    # load params
    model_mergebn = load_params(model_mergebn, params)
    top1_acc, top5_acc = test(model_mergebn, testLoader, topk=(1, 5) if args.data_set == 'imagenet' else (1, ))

    model_state_dict = model_mergebn.state_dict()

    state = {
        'state_dict': model_state_dict,
        'weight_fl': weight,
        'bias_fl': bias,
        'a_fl' : a_fl,
    }
    checkpoint.save_model(state, is_best=True)
    logger.info('Quantization Top-1 accuracy: {:.3f} Top-5 accuracy: {:.3}'.format(float(top1_acc), float(top5_acc)))


    del(model_baseline)

if __name__ == '__main__':
    main()