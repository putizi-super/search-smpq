import os
import math
import random
import numpy as np
import sys
import argparse

import torch
import torch.nn as nn

sys.path.append('../../')
from utils.options import args
from tool.meter import *
import tool.pruning as tp

if args.data_set == "cifar10":
    import models.cifar10 as models
elif args.data_set == "imagenet":
    import models.imagenet as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def softmax_numpy(x):
    e_x = np.exp(x.astype(float) - np.max(x))
    return e_x / e_x.sum()


def uniform_normal_init(loc, scale, size):
    a = np.random.normal(loc=loc,scale=scale,size=size)
    mask = a <= loc
    b = np.random.uniform(0.001,loc,size)
    out = a*(1-mask)+b*mask
    return out

def mix_normal_init(loc, size, max_value):
    out = []
    scale1 = (max_value-loc)/3.0
    scale2 = (loc-0.001)/3.0
    cnt = 0
    while cnt < size:
        a = np.random.normal(loc=loc,scale=scale1)
        if a > 0.1:
            a = max_value if (a > max_value) else a
            out.append(a)
        else:
            while True:
                b=np.random.normal(loc=loc,scale=scale2)
                if b < 0.1 and b>0:
                    out.append(b)
                    break
        cnt+=1
    out = np.array(out)
    return out

def judgment_value(args,a,min,max):
    m1 = a < min
    # a = (0.5/args.max_PARAMs)*m1+a*(1-m1)
    a = min*m1+a*(1-m1)
    m2 = a > max
    a = max*m2+a*(1-m2)
    return a

def filter_prune(model, example_inputs, output_transform, pruning_rates):
    model.cpu().eval()
    prunable_module_type = (nn.Conv2d)
    prunable_modules = [ m for m in model.modules() if isinstance(m, prunable_module_type) ]
    DG = tp.DependencyGraph().build_dependency( model, example_inputs=example_inputs, output_transform=output_transform )

    for layer_to_prune, fp_rate in zip(prunable_modules,pruning_rates):
        if isinstance( layer_to_prune, nn.Conv2d ):
            prune_fn = tp.prune_conv
        weight = layer_to_prune.weight.detach().cpu().numpy()
        out_channels = weight.shape[0]
        num_pruned = int(out_channels * fp_rate)
        rand_idx = random.sample( list(range(out_channels)),  num_pruned )
        plan = DG.get_pruning_plan( layer_to_prune, prune_fn, rand_idx)
        plan.exec()

    with torch.no_grad():
        out = model( example_inputs )
        if output_transform:
            out = output_transform(out)
    return model

def random_can_fp(args, input_size, num, num_states, total_flops, total_params):
    print('random select ........', flush=True)
    candidates = []
    candidates_int = []
    while(len(candidates))<num:
        if args.max_FLOPs_FP == 0 and args.max_PARAMs_FP == 0:
            can = np.zeros(num_states+2)
            can[-1] = total_flops
            can[-2] = total_params            
        else:
            # # uniform random init
            # high = 2.0/max(args.max_PARAMs_FP,args.max_FLOPs_FP)
            # can = np.random.uniform(0.01,high,num_states+2)

            # ## uniform normal init
            # loc = 1/max(args.max_PARAMs,args.max_FLOPs)
            # scale = (0.6-loc)/3.0
            # can = uniform_normal_init(loc,scale,num_states+2)

            # # # mix normal_init
            # loc = 1.0/max(args.max_PARAMs_FP,args.max_FLOPs_FP)
            # can = mix_normal_init(loc,num_states+2,0.99)

            # # gamma init
            loc = max(args.max_PARAMs_FP, args.max_FLOPs_FP)
            can = np.random.gamma(loc*8.0, 0.125, num_states + 2)
            mask1 = can >= 0.9999
            mask2 = can <= 0.0001
            can = mask1 * 0.9999 + (1 - mask1) * can
            can = mask2 * 0.0001 + (1 - mask2) * can

            t_can = tuple(can[:-2])
            # print(can[:-2].tolist())

            model = models.__dict__[args.arch]()

            fp_model = filter_prune(model,torch.randn(1,3,input_size,input_size),output_transform=None,pruning_rates=t_can)

            layer_flops = calc_model_flops(fp_model,input_size,mul_add=False)
            sparse_total_flops = sum(layer_flops)
            layer_params = calc_model_parameters(fp_model)
            sparse_total_params = sum(layer_params)
            # print(1.0-sparse_total_flops/total_flops)
            # print(1.0-sparse_total_params/total_params)

            if args.max_FLOPs_FP == 0 and args.max_PARAMs_FP !=0:
                if 1.0-sparse_total_params/total_params < args.target_params_fp or 1.0-sparse_total_params/total_params > args.target_params_fp+0.1:
                    continue
            elif args.max_FLOPs_FP != 0 and args.max_PARAMs_FP ==0:
                if 1.0-sparse_total_flops/total_flops < args.target_flops_fp or 1.0-sparse_total_flops/total_flops > args.target_flops_fp+0.1:
                    continue
            else:
                if 1.0-sparse_total_flops/total_flops < args.target_flops_fp or 1.0-sparse_total_flops/total_flops > args.target_flops_fp+0.1 or 1.0-sparse_total_params/total_params < args.target_params_fp or 1.0-sparse_total_params/total_params > args.target_params_fp+0.1:
                    continue

            can[-1] = sparse_total_flops
            can[-2] = sparse_total_params

            # compare difference
            t_can_int = [math.ceil(i*100.0) for i in t_can]


            if t_can_int in candidates_int:
                continue
            else:
                candidates_int.append(t_can_int)

        # print(total_flops/sparse_total_flops)
        print(len(candidates))

        candidates.append(can.tolist())
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates

# mutation突变 operation in evolution algorithm
def get_mutation_fp(args, input_size, epoch, keep_top_k, top_k_acc,num_states, mutation_num, m_prob, strength, total_flops,total_params):
    print('mutation ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for g_can in keep_top_k:
        global_candidates_int.append([math.ceil(i*100.0) for i in g_can[:-2]])
    k = len(keep_top_k)
    iter = 0
    max_iters = 10*mutation_num
    top_k_acc = top_k_acc / 100.0
    top_k_p = softmax_numpy(top_k_acc)
    # while len(res)<mutation_num and iter<max_iters:
    while len(res)<mutation_num:

        if args.max_FLOPs_FP == 0 and args.max_PARAMs_FP == 0:
            can = np.zeros(num_states+2)
            can[-1] = total_flops
            can[-2] = total_params     
            res.append(can.tolist())       
        else:
            # ids = np.random.choice(k, mutation_num,replace=False,p=top_k_p)
            # ids = np.random.choice(k, mutation_num,replace=False)
            # ids = np.random.choice(k, mutation_num,p=top_k_p)
            ids = np.random.choice(k, mutation_num)
            # print(ids)
            select_seed = np.array([keep_top_k[id] for id in ids])
            # if epoch < 50:
            #     alpha = np.random.uniform(0,1,num_states+2)
            # elif epoch < 100:
            #     alpha = np.random.uniform(0.25,0.75,num_states+2)
            # elif epoch < 120:
            #     alpha = np.random.uniform(0.35,0.65,num_states+2)
            # alpha = np.random.uniform(0,1,num_states+2)
            alpha = np.random.normal(loc=0.5,scale=0.1,size=num_states+2)
            is_m = np.random.choice(np.arange(0,2), (mutation_num, num_states+2), p=[1-m_prob, m_prob]).astype(np.float32)
            # is_m[:,0] = np.random.choice(np.arange(0,2),len(is_m[:,0]) , p=[0.6, 0.4]).astype(np.float32)
            # is_m[:,-1] = np.random.choice(np.arange(0,2),len(is_m[:,-1]) , p=[0.6, 0.4]).astype(np.float32)
            # select_list =  select_seed*(1.0-is_m)+(1.0-select_seed)*is_m
            mask1 = alpha >= 1
            mask2 = alpha <= 0
            alpha = alpha*(1-mask1)+0.5*(mask1) 
            alpha = alpha*(1-mask2)+0.5*(mask2)
            mask = alpha < 0.5
            beta = (pow((2*alpha),(1/strength))-1)*mask + (1-pow((2*(1-alpha)),(1/strength)))*(1-mask)
            select_list =  select_seed+beta*is_m
            select_list = judgment_value(args,select_list,0.0001,0.9999)
            iter += 1
            cnt = 0
            for can in select_list:
                cnt = cnt + 1
                sum_mask = sum(is_m[cnt-1])
                t_can = tuple(can[:-2])

                model = models.__dict__[args.arch]()

                fp_model = filter_prune(model,torch.randn(1,3,input_size,input_size),output_transform=None,pruning_rates=t_can)

                layer_flops = calc_model_flops(fp_model,input_size,mul_add=False)
                sparse_total_flops = sum(layer_flops)
                layer_params = calc_model_parameters(fp_model)
                sparse_total_params = sum(layer_params)

                if args.max_FLOPs_FP == 0 and args.max_PARAMs_FP !=0:
                    if 1.0-sparse_total_params/total_params < args.target_params_fp or 1.0-sparse_total_params/total_params > args.target_params_fp+0.1:
                        continue
                elif args.max_FLOPs_FP != 0 and args.max_PARAMs_FP ==0:
                    if 1.0-sparse_total_flops/total_flops < args.target_flops_fp or 1.0-sparse_total_flops/total_flops > args.target_flops_fp+0.1:
                        continue
                else:
                    if 1.0-sparse_total_flops/total_flops < args.target_flops_fp or 1.0-sparse_total_flops/total_flops > args.target_flops_fp+0.1 or 1.0-sparse_total_params/total_params < args.target_params_fp or 1.0-sparse_total_params/total_params > args.target_params_fp+0.1:
                        continue
                can[-1] = sparse_total_flops
                can[-2] = sparse_total_params
                # print(total_flops/sparse_total_flops)
                # print(total_params/sparse_total_params)

                # compare difference
                t_can_int = [math.ceil(i*100.0) for i in t_can]
                if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                    continue
                else:
                    candidates_int.append(t_can_int)

                res.append(can)
                if len(res)==mutation_num:
                    break
    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

# crossover交叉 operation in evolution algorithm
def get_crossover_fp(args,input_size, keep_top_k, top_k_acc, num_states, crossover_num, total_flops,total_params):
    print('crossover ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for g_can in keep_top_k:
        global_candidates_int.append([math.ceil(i*100.0) for i in g_can[:-2]])
    k = len(keep_top_k)
    # top_k_acc = top_k_acc / 100.0
    top_k_p = softmax_numpy(top_k_acc)
    iter = 0
    max_iters = 10 * crossover_num
    # while len(res)<crossover_num and iter<max_iters:
    while len(res)<crossover_num:

        if args.max_FLOPs_FP == 1 and args.max_PARAMs_FP == 1:
            can = np.zeros(num_states+2)
            can[-1] = total_flops
            can[-2] = total_params     
            res.append(can.tolist())       
        else:
            # id1, id2 = np.random.choice(k, 2, replace=False)
            # id1, id2 = np.random.choice(k, 2, replace=False,p=top_k_p)
            # print([id1,id2])
            # p1 = keep_top_k[id1]
            # p2 = keep_top_k[id2]
            id1 = np.random.choice(k, 1, p=top_k_p)
            id2 = np.random.choice(k, 1)
            # print([id1[0],id2[0]])
            p1 = keep_top_k[id1[0]]
            p2 = keep_top_k[id2[0]]

            # recombination
            alpha = np.random.rand(len(p1))
            can = p1*alpha + p2*(1.0-alpha)

            # alpha = np.random.uniform(-1, 1, len(p1))
            # can = p1+alpha*(p1-p2)

            can = judgment_value(args,can,0.0001,0.9999)
            # ## Discrete recombination
            # mask = np.random.randint(low=0, high=2, size=(num_states+2)).astype(np.float32)
            # can = p1*mask + p2*(1.0-mask)
            iter += 1
            t_can = tuple(can[:-2])

            model = models.__dict__[args.arch]()

            fp_model = filter_prune(model,torch.randn(1,3,input_size,input_size),output_transform=None,pruning_rates=t_can)

            layer_flops = calc_model_flops(fp_model,input_size,mul_add=False)
            sparse_total_flops = sum(layer_flops)
            layer_params = calc_model_parameters(fp_model)
            sparse_total_params = sum(layer_params)

            if args.max_FLOPs_FP == 0 and args.max_PARAMs_FP !=0:
                if 1.0-sparse_total_params/total_params < args.target_params_fp or 1.0-sparse_total_params/total_params > args.target_params_fp+0.1:
                    continue
            elif args.max_FLOPs_FP != 0 and args.max_PARAMs_FP ==0:
                if 1.0-sparse_total_flops/total_flops < args.target_flops_fp or 1.0-sparse_total_flops/total_flops > args.target_flops_fp+0.1:
                    continue
            else:
                if 1.0-sparse_total_flops/total_flops < args.target_flops_fp or 1.0-sparse_total_flops/total_flops > args.target_flops_fp+0.1 or 1.0-sparse_total_params/total_params < args.target_params_fp or 1.0-sparse_total_params/total_params > args.target_params_fp+0.1:
                    continue

            can[-1] = sparse_total_flops
            can[-2] = sparse_total_params
            # print(total_flops/sparse_total_flops)
            # print(total_params/sparse_total_params)

            # compare difference
            t_can_int = [math.ceil(i*100.0) for i in t_can]
            if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                continue
            else:
                candidates_int.append(t_can_int)

            res.append(can)
            if len(res)==crossover_num:
                break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res
