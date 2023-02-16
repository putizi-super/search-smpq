import os
import math
import random
import numpy as np
import sys
import argparse

def softmax_numpy(x):
    e_x = np.exp(x.astype(float) - np.max(x))
    return e_x / e_x.sum()

def print_model_parm_flops(total_flops,total_params,flops, params, sparse_rate):
    sparse_flops = [(1.0-a)*b for a,b in zip(sparse_rate,flops)]
    sparse_params = [(1.0-a)*b for a,b in zip(sparse_rate,params)]
    sparse_total_flops = total_flops - sum(sparse_flops)
    sparse_total_params = total_params - sum(sparse_params)
    return sparse_total_flops,sparse_total_params

def uniform_normal_init(loc, scale, size):
    a = np.random.normal(loc=loc,scale=scale,size=size)
    mask = a <= loc
    b = np.random.uniform(0.001,loc,size)
    out = a*(1-mask)+b*mask
    return out

def mix_normal_init(loc, size, max_value):
    out = []
    scale1 = (max_value - loc) / 3.0
    scale2 = (loc - 0.001) / 3.0
    cnt = 0
    while cnt < size:
        a = np.random.normal(loc=loc, scale=scale1)
        if a > 0.1:
            a = max_value if (a > max_value) else a
            out.append(a)
        else:
            while True:
                b=np.random.normal(loc=loc, scale=scale2)
                if b < 0.1 and b > 0:
                    out.append(b)
                    break
        cnt += 1
    out = np.array(out)
    return out

def gen_prob_4_ACO(tau, alpha):
    p = []
    for _tau in tau:
        pi = _tau**alpha / np.sum(_tau**alpha)
        p.append(pi)
    return np.array(p)

def judgment_value(args, a, min, max):
    m1 = a < min
    a = (0.5 / args.max_PARAMs) * m1 + a * (1 - m1)
    m2 = a > max
    a = max * m2 + a * (1 - m2)
    return a

def random_can(args, ants_num, num_states, nodes_num, tau0, flops, params, total_flops, total_params):
    print('random select ........', flush=True)
    nodes = np.linspace(0, 1, nodes_num)
    tau = np.full(shape=[num_states, nodes_num - 1], fill_value=tau0)
    candidates = []
    while(len(candidates)) < num:
        # # uniform random init
        # high = 2.0/max(args.max_PARAMs,args.max_FLOPs)
        # can = np.random.uniform(0.001,high,num_states+2)

        # # uniform normal init
        # loc = 1/max(args.max_PARAMs,args.max_FLOPs)
        # scale = (0.6-loc)/3.0
        # can = uniform_normal_init(loc,scale,num_states+2)

        # # mix normal_init
        # loc = 1.0 / max(args.max_PARAMs, args.max_FLOPs)
        # can = mix_normal_init(loc, num_states + 2, 0.99)

        # gamma init
        loc = 1.0 / max(args.max_PARAMs, args.max_FLOPs)
        can = np.random.gamma(loc * 10, 0.1, size=num_states + 2)
        mask = can > 0.99
        can = mask * 0.99 + (1 - mask) * can

        sparse_total_flops, sparse_total_params = print_model_parm_flops(total_flops, total_params, flops, params, can[1:-2])
        if (total_flops / sparse_total_flops < args.max_FLOPs) or (total_params / sparse_total_params < args.max_PARAMs):
            continue
        can[-1] = sparse_total_flops
        can[-2] = sparse_total_params
        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates, can, nodes

def get_mutation(args, epoch, keep_top_k, top_k_acc,num_states, mutation_num, m_prob, strength, flops,params,total_flops,total_params):
    print('mutation ......', flush=True)
    res = []
    int_res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10 * mutation_num
    top_k_acc = top_k_acc / 100.0
    top_k_p = softmax_numpy(top_k_acc)
    # while len(res)<mutation_num and iter<max_iters:
    while len(res) < mutation_num:
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
        alpha = np.random.normal(loc=0.5, scale=0.1, size=num_states + 2)
        is_m = np.random.choice(np.arange(0, 2), (mutation_num, num_states + 2), p=[1-m_prob, m_prob]).astype(np.float32)
        # is_m[:,0] = np.random.choice(np.arange(0,2),len(is_m[:,0]) , p=[0.6, 0.4]).astype(np.float32)
        # is_m[:,-1] = np.random.choice(np.arange(0,2),len(is_m[:,-1]) , p=[0.6, 0.4]).astype(np.float32)
        # select_list =  select_seed*(1.0-is_m)+(1.0-select_seed)*is_m
        mask1 = alpha >= 1
        mask2 = alpha <= 0
        alpha = alpha * (1 - mask1) + 0.5 * (mask1) 
        alpha = alpha * (1 - mask2) + 0.5 * (mask2)
        mask = alpha < 0.5
        beta = (pow((2*alpha), (1/strength))-1) * mask + (1 - pow((2 * (1 - alpha)), (1 / strength))) * (1 - mask)
        select_list =  select_seed + beta * is_m
        select_list = judgment_value(args, select_list, 0.001, 0.99)
        iter += 1
        cnt = 0
        for can in select_list:
            cnt = cnt + 1
            sum_mask = sum(is_m[cnt-1])
            t_can = tuple(can[:-2])
            sparse_total_flops, sparse_total_params = print_model_parm_flops(total_flops, total_params, flops, params, can[:-2])
            if (total_flops / sparse_total_flops < args.max_FLOPs) or (total_params / sparse_total_params < args.max_PARAMs) or sum_mask==0:
                continue
            can[-1] = sparse_total_flops
            can[-2] = sparse_total_params
            # print(total_flops/sparse_total_flops)
            # print(total_params/sparse_total_params)
            res.append(can)
            if len(res) == mutation_num:
                break
    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

def get_crossover(args, top_k_ants, top_k_acc, num_states, alpha, tau, nodes, Q, rho, flops, params, total_flops, total_params):
    print('crossover ......', flush=True)
    tau = (1 - rho) * tau
    for i, ant in enumerate(top_k_ants):
        for layernum, p_rate in enumerate(ant[:-2]):
            intvl_id = np.where(p_rate < nodes)
            intvl_id = intvl_id[0][0] - 1
            tau[i, intvl_id] += Q / (1 - top_k_acc[i])
    
    for i, ant in enumerate(top_k_ants):
        p_mat = gen_prob_4_ACO(tau, alpha)
        while True:
            for j in range(num_states):
                low_id = np.random.choice(range(len(nodes) - 1) , p=p_mat[j])
                ant[j] = np.random.uniform(nodes[low_id], nodes[low_id + 1])
            quan_total_flops, quan_total_params = print_model_parm_flops(flops, params, ant[:-2])
            if (total_flops/sparse_total_flops < args.max_FLOPs) or (total_params/sparse_total_params < args.max_PARAMs):
                continue
            ant[-1] = quan_total_flops
            ant[-2] = quan_total_params
            break

    return top_k_ants

















    top_k_p = softmax_numpy(top_k_acc)

    # while len(res)<crossover_num and iter<max_iters:
    while len(res)<crossover_num:
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
        can = judgment_value(args,can,0.001,0.99)
        # ## Discrete recombination
        # mask = np.random.randint(low=0, high=2, size=(num_states+2)).astype(np.float32)
        # can = p1*mask + p2*(1.0-mask)

        t_can = tuple(can[:-2])
        sparse_total_flops, sparse_total_params= print_model_parm_flops(total_flops,total_params,flops,params,can[:-2])
        if (total_flops/sparse_total_flops < args.max_FLOPs) or (total_params/sparse_total_params < args.max_PARAMs):
            continue
        can[-1] = sparse_total_flops
        can[-2] = sparse_total_params
        # print(total_flops/sparse_total_flops)
        # print(total_params/sparse_total_params)
        res.append(can)
        if len(res)==crossover_num:
            break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res
