import os
import math
import random
import numpy as np
import sys
import argparse
sys.path.append('..')
from data import cifar10
from  modules.search_sq.quantization import *


def bw_softmax_numpy(x):
    e_x = np.exp(x.astype(float) - np.max(x))
    return e_x / e_x.sum()

def gen_prob_4_ACO(tau, alpha):
    p = []
    for _tau in tau:
        pi = _tau**alpha / np.sum(_tau**alpha)
        p.append(pi)
    return np.array(p)

def print_model_parm_flops(flops, params, bw):
    quan_flops = [a * b for a, b in zip(bw, flops)]
    quan_params = [a * b for a, b in zip(bw, params)]
    quan_total_flops = sum(quan_flops)
    quan_total_params = sum(quan_params)
    return quan_total_flops,quan_total_params

def bw_random_can(args, num, num_states, tau0, flops, params, total_flops, total_params):
    print('random select ........', flush=True)
    tau = np.full(shape=[num_states, args.max_bw], fill_value=tau0)
    candidates = []
    while(len(candidates)) < num:
        # # uniform random init
        can = np.random.randint(1, args.max_bw + 1, num_states + 2)
        # normal_init
        # can =  np.round(np.random.normal(loc=0,scale=1,size=num_states))

        quan_total_flops, quan_total_params = print_model_parm_flops(flops, params, can[:-2])
        if  quan_total_params / total_params >= args.avg_bw:
            continue
        can[-1] = quan_total_flops
        can[-2] = quan_total_params

        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates, tau

def bw_get_mutation(args,epoch, keep_top_k, top_k_acc,num_states, mutation_num, m_prob,strength,max_range,flops, params, total_flops, total_params):
    print('mutation ......', flush=True)
    res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10*mutation_num
    top_k_acc = top_k_acc / 100.0
    top_k_p = bw_softmax_numpy(top_k_acc)
    while len(res)<mutation_num and iter<max_iters:
    # while len(res)<mutation_num:
        # ids = np.random.choice(k, mutation_num,replace=False,p=top_k_p)
        # ids = np.random.choice(k, mutation_num,replace=False)
        # ids = np.random.choice(k, mutation_num,p=top_k_p)
        ids = np.random.choice(k, mutation_num)
        # print(ids)
        select_seed = np.array([keep_top_k[id] for id in ids])

        is_m = np.random.choice(np.arange(0,2), (mutation_num, num_states+2), p=[1-m_prob, m_prob])

        alpha = np.random.uniform(0,1,num_states+2)
        # alpha = np.random.normal(loc=0.5,scale=0.1,size=num_states)
        mask1 = alpha >= 1
        mask2 = alpha <= 0
        alpha = alpha * (1 - mask1) + 0.5 * (mask1) 
        alpha = alpha * (1 - mask2) + 0.5 * (mask2)
        mask = alpha < 0.5
        beta = np.round(max_range*((pow((2*alpha),(1/strength))-1)*mask + (1-pow((2*(1-alpha)),(1/strength)))*(1-mask)))
        # select_list =  select_seed*(1-is_m) + select_seed*is_m*(-1)
        # print('-----------------')
        # print(beta)
        # print(beta*is_m)
        # print('-----------------')
        select_list =  select_seed+beta*is_m

        m = select_list <= 0
        select_list = select_list*(1-m) + m
 
        iter += 1
        cnt = 0
        for can in select_list:

            # t_can = tuple(can[:-2])
            mask = can[:-2] > args.max_bw
            can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
            quan_total_flops, quan_total_params= print_model_parm_flops(flops,params,can[:-2])
            if  quan_total_params/total_params >= args.avg_bw:
                continue
            can[-1] = quan_total_flops
            can[-2] = quan_total_params

            if (can.tolist() not in res) and (can.tolist() not in keep_top_k):
                cnt = cnt + 1
                res.append(can.tolist())
                if len(res)==mutation_num:
                    break
    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

def bw_get_crossover(args, top_k_ants, top_k_acc, num_states, alpha, tau, Q, rho, flops, params, total_flops, total_params):
    print('crossover ......', flush=True)
    tau = (1 - rho) * tau
    for i, ant in enumerate(top_k_ants):
        for layer_num, bw in enumerate(ant[:-2]):
            tau[layer_num][int(bw) - 1] += Q / (1 - top_k_acc[i])
    
    for i, ant in enumerate(top_k_ants):
        p_mat = gen_prob_4_ACO(tau, alpha)
        while True:
            for j in range(num_states):
                ant[j] = np.random.choice(range(1, args.max_bw + 1) , p=p_mat[j])
            quan_total_flops, quan_total_params = print_model_parm_flops(flops, params, ant[:-2])
            if  quan_total_params / total_params >= args.avg_bw:
                continue
            ant[-1] = quan_total_flops
            ant[-2] = quan_total_params
            break
    return top_k_ants
