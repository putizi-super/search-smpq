import argparse
import ast

parser = argparse.ArgumentParser(description='Filter Pruning')

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    help='Select dataset to train. default:cifar10',
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/home/ic619/xk/pytorch/data/cifar-10/',
    help='The dictionary where the input is stored. default:/home/ic619/xk/pytorch/data/cifar-10/',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments')

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='resnet',
    help='Architecture of model. default:resnet')

parser.add_argument(
    '--arch_teacher',
    type=str,
    default='resnet',
    help='Architecture of model. default:resnet')

parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56',
    help='Detail architecuture of model. default:resnet56'
)

parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="the type of the model used, e.g. \"resnet\", \"googlenet\", ..."
)

parser.add_argument(
    '--optimizer',
    type=str,
    default='SGD',
    help='Training optimizer. default:SGD'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,
    help='The num of epochs to train. default:150')

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training. default:128')

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation. default:100')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='Learning rate for train. default:1e-2'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[50, 100],
    help='the iterval of learn rate. default:50, 100'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='The weight decay of loss. default:5e-4')

# parser.add_argument(
#     '--warmup', 
#     dest='warmup', 
#     type=ast.literal_eval,
#     help='True or False warmup,input should be either True or False'
# )

parser.add_argument(
    '--warmup_epochs',
    type=int,
    default=5,
    help='warmup epochs for training. default:5')

parser.add_argument(
    '--label-smoothing',
    default=0.0,
    type=float,
    metavar='S',
    help='label smoothing')

# parser.add_argument(
#     '--baseline_model',
#     type=str,
#     default='./baseline/imagenet/resnet18.pth',
#     # default='./baseline/imagenet/darknet19.pth',
#     # default='./baseline/imagenet/regnet_y_400.pth',
#     help='Path to the model wait for test. default:None'
# )
parser.add_argument(
    '--baseline_model',
    type=str,
    default='./baseline/imagenet/darknet19.pth',
    help='Path to the model wait for test. default:None'
)

parser.add_argument(
    '--teacher_model',
    type=str,
    default=None,
    help='Path to the model wait for test. default:None'
)

parser.add_argument(
    '--baseline', 
    dest='baseline', 
    type=ast.literal_eval,
    help='True or False fine_tune,input should be either True or False'
)

parser.add_argument(
    '--resume_model',
    type=str,
    default=None,
    help='Path to the model wait for test. default:None'
)

parser.add_argument(
    '--resume', 
    dest='resume', 
    type=ast.literal_eval,
    help='True or False fine_tune,input should be either True or False'
)

parser.add_argument(
    '--workers',
    type=int,
    default=16,
    help='workers number. default:16'
)

## data augument

parser.add_argument(
    '--cutout', 
    dest='cutout', 
    type=ast.literal_eval,
    help='True or False cutout'
)

parser.add_argument(
    '--cutout_length',
    type=int,
    default=16,
    help='cutout length. default:16'
)

parser.add_argument(
    '--autoaug', 
    dest='autoaug', 
    type=ast.literal_eval,
    help='True or False autoaug'
)

parser.add_argument(
    '--mixup',
    default=0.0,
    type=float,
    metavar='ALPHA',
    help='mixup alpha'
)

## Struct Pruning
parser.add_argument(
    '-t_f_fp', 
    '--target_flops_fp', 
    default=0.5, 
    type=float,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '-t_p_fp', 
    '--target_params_fp', 
    default=0.5, 
    type=float,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '-m_f_fp', 
    '--max_FLOPs_FP', 
    default=8, 
    type=float,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '-m_p_fp', 
    '--max_PARAMs_FP', 
    default=8, 
    type=float,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '-fp_rate_len', 
    '--FP_Rate_Len', 
    default=8, 
    type=int,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '--schedule', 
    type=int, 
    nargs='+', 
    default=[150, 225],
    help='Decrease learning rate at these epochs.'
)

parser.add_argument(
    '--gammas', 
    type=float, 
    nargs='+', 
    default=[0.1, 0.1],
    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)

## Unstruct Pruning
parser.add_argument(
    '--start_conv',
    type=int,
    default=1,
    help='The index of Conv to start pruning, index starts from 0. default:1'
)

parser.add_argument(
    '--pruning_rate',
    type=str,
    default=None,
    help='The proportion of each layer reserved after pruning convolution layer. default:None'
)

parser.add_argument(
    '--pruning_model',
    type=str,
    default=None,
    help='Path to the model wait for Pruning/test. default:None'
)


parser.add_argument(
    '-m_f', 
    '--max_FLOPs', 
    default=8, 
    type=float,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '-m_p', 
    '--max_PARAMs', 
    default=8, 
    type=float,
    metavar='N', 
    help='set max pruning FLOPs time'
)

parser.add_argument(
    '--no_fc', 
    dest='no_fc', 
    type=ast.literal_eval,
    help='True or False no fc'
)

## Quantization

parser.add_argument(
    '--max_bw', 
    default=8, 
    type=int, 
    metavar='N', 
    help='max bit-width for search'
)

parser.add_argument(
    '--min_bw', 
    default=1, 
    type=int, 
    metavar='N', 
    help='min bit-width for search'
)

parser.add_argument(
    '--avg_bw_weights', 
    default=3.0, 
    type=float, 
    metavar='N', 
    help='avg bit-width for search'
)

parser.add_argument(
    '--fix_bw_weights', 
    dest='fix_bw_weights', 
    type=ast.literal_eval,
    help='True or False fix bw weights'
)

parser.add_argument(
    '--avg_bw_fm', 
    default=3.0, 
    type=float, 
    metavar='N', 
    help='number of total epochs to run'
)

parser.add_argument(
    '--fix_bw_fm', 
    dest='fix_bw_fm', 
    type=ast.literal_eval,
    help='True or False fix bw featrue map'
)

parser.add_argument(
    '--search_quan_type', 
    type=str, 
    default='tensorrt', 
    help='quantization type name'
)

## search

parser.add_argument(
    '--search_epochs',
    type=int,
    default=40,
    help='search epoch. default:40')

parser.add_argument(
    '--search_step', 
    dest='search_step',
    type=ast.literal_eval,
    help='True or False search_step,input should be either True or False'
)

parser.add_argument(
    '--training_step', 
    dest='training_step', 
    type=ast.literal_eval,
    help='True or False training_step,input should be either True or False'
)

parser.add_argument(
    '--fine_tune', 
    dest='fine_tune', 
    type=ast.literal_eval,
    help='True or False fine_tune,input should be either True or False'
)

## Data-Free
parser.add_argument(
    '--generator_baseline_model',
    type=str,
    default=None,
    help='Path to the model wait for test. default:None'
)

parser.add_argument(
    '--generator_baseline', 
    dest='generator_baseline', 
    type=ast.literal_eval,
    help='True or False fine_tune,input should be either True or False'
)

## Wandb
parser.add_argument(
    '--wandb', 
    dest='wandb', 
    type=ast.literal_eval,
    help='True or False no fc'
)

parser.add_argument(
    '--wandb_prj',
    type=str,
    default="Test",
    help='Path to the model wait for test. default:None'
)

# dali
parser.add_argument(
    '--dali', 
    dest='dali', 
    type=ast.literal_eval,
    help='True or False using dali'
)

### 添加
parser.add_argument(
    '--w_bit', 
    type=int,
    default=4,
    help='the bits of weight to quant'
)
parser.add_argument(
    '--a_bit', 
    type=int,
    default=8,
    help='the bits of activation to quant'
)

parser.add_argument(
    '--channel_wise', 
    type=bool,
    default=False,
    help='true or false by channel_wise'
)


# args = parser.parse_args()