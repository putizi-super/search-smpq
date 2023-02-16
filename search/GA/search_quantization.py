import os
import math
import random
import numpy as np
import sys
import argparse


def bw_softmax_numpy(x):
    e_x = np.exp(x.astype(float) - np.max(x))
    return e_x / e_x.sum()

def print_model_parm_flops(flops, params, bw):
    quan_flops = [a*b for a,b in zip(bw,flops)]
    quan_params = [a*b for a,b in zip(bw,params)]
    quan_total_flops = sum(quan_flops)
    quan_total_params = sum(quan_params)
    return quan_total_flops,quan_total_params

def print_model_featuremap(featuremaps,bw):
    quan_featuremaps = [a*b for a,b in zip(bw,featuremaps)]
    quan_total_featuremaps = sum(quan_featuremaps)
    return quan_total_featuremaps

# random select operation in evolution algorithm
def bw_weights_random_can(args,num, num_states,flops, params, total_flops, total_params):
    print('random select ........', flush=True)
    candidates = []
    while(len(candidates))<num:
        if args.fix_bw_weights == True:
            can = np.ones(num_states+2)*args.avg_bw_weights
            can[-1] = total_flops*args.avg_bw_weights
            can[-2] = total_params*args.avg_bw_weights
        else:
            # # uniform random init
            # can = np.random.randint(1,np.ceil(args.avg_bw_weights*2),num_states+2) #1.5
            can = np.random.randint(args.min_bw,args.max_bw,num_states+2) #1.5
            can[0] = np.ceil(args.avg_bw_weights*3) if np.ceil(args.avg_bw_weights*3) < args.max_bw else args.max_bw 
            can[-3] = np.ceil(args.avg_bw_weights*3) if np.ceil(args.avg_bw_weights*3) < args.max_bw else args.max_bw 
            # can = np.random.randint(1,args.max_bw,num_states+2) #1.5
            # normal_init
            # can =  np.round(np.random.normal(loc=0,scale=1,size=num_states))

            mask = can[:-2] > args.max_bw
            can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
            mask = can[:-2] < args.min_bw
            can[:-2] = can[:-2]*(1-mask)+args.min_bw*mask
            quan_total_flops, quan_total_params= print_model_parm_flops(flops,params,can[:-2])
            if  quan_total_params/total_params >= args.avg_bw_weights:
                continue
            can[-1] = quan_total_flops
            can[-2] = quan_total_params

        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates

# mutation operation in evolution algorithm
def bw_weights_get_mutation(args,epoch, keep_top_k, top_k_acc,num_states, mutation_num, m_prob,strength,max_range,flops, params, total_flops, total_params):
    print('mutation ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for g_can in keep_top_k:
        global_candidates_int.append([i for i in g_can[:-2]])
    k = len(keep_top_k)
    iter = 0
    max_iters = 10*mutation_num
    # top_k_acc = top_k_acc / 100.0
    top_k_p = bw_softmax_numpy(top_k_acc)
    # while len(res)<mutation_num and iter<max_iters:
    while len(res)<mutation_num:
        if args.fix_bw_weights == True:
            can = np.ones(num_states+2)*args.avg_bw_weights
            can[-1] = total_flops*args.avg_bw_weights
            can[-2] = total_params*args.avg_bw_weights
            res.append(can.tolist())
        else:
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
            alpha = alpha*(1-mask1)+0.5*(mask1) 
            alpha = alpha*(1-mask2)+0.5*(mask2)
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
                
                mask = can[:-2] > args.max_bw
                can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
                mask = can[:-2] < args.min_bw
                can[:-2] = can[:-2]*(1-mask)+args.min_bw*mask
                t_can = tuple(can[:-2])

                quan_total_flops, quan_total_params= print_model_parm_flops(flops,params,can[:-2])
                if  quan_total_params/total_params >= args.avg_bw_weights:
                    continue
                can[-1] = quan_total_flops
                can[-2] = quan_total_params

                # compare difference
                t_can_int = [i for i in t_can]
                if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                    continue
                else:
                    candidates_int.append(t_can_int)

                if (can.tolist() not in res) and (can.tolist() not in keep_top_k):
                    cnt = cnt + 1
                    res.append(can.tolist())
                    if len(res)==mutation_num:
                        break
    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

# crossover operation in evolution algorithm
def bw_weights_get_crossover(args,keep_top_k, top_k_acc, num_states, crossover_num,flops, params, total_flops, total_params):
    print('crossover ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for g_can in keep_top_k:
        global_candidates_int.append([i for i in g_can[:-2]])
    k = len(keep_top_k)
    # top_k_acc = top_k_acc / 100.0
    top_k_p = bw_softmax_numpy(top_k_acc)
    iter = 0
    max_iters = 10 * crossover_num
    while len(res)<crossover_num and iter<max_iters:
        if args.fix_bw_weights == True:
            can = np.ones(num_states+2)*args.avg_bw_weights
            can[-1] = total_flops*args.avg_bw_weights
            can[-2] = total_params*args.avg_bw_weights
            res.append(can.tolist())
        else:
            # id1, id2 = np.random.choice(k, 2, replace=False,p=top_k_p)
            # id1, id2 = np.random.choice(k, 2, replace=False)
            # print([id1,id2])
            # p1 = keep_top_k[id1]
            # p2 = keep_top_k[id2]
            id1 = np.random.choice(k, 1, p=top_k_p)
            id2 = np.random.choice(k, 1)
            # print([id1[0],id2[0]])
            p1 = keep_top_k[id1[0]]
            p2 = keep_top_k[id2[0]]
            
            ## recombination

            # alpha = np.random.rand(len(p1))
            # p1 = np.array(p1)
            # p2 = np.array(p2)
            # can = np.round(p1*alpha + p2*(1.0-alpha))

            alpha = np.random.uniform(-1, 1, len(p1))
            p1 = np.array(p1)
            p2 = np.array(p2)
            can = np.round((1 + alpha) * p1 - alpha * p2)

            m = can <= 0
            can = can*(1.0-m)+m

            mask = can[:-2] > args.max_bw
            can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
            mask = can[:-2] < args.min_bw
            can[:-2] = can[:-2]*(1-mask)+args.min_bw*mask
            t_can = tuple(can[:-2])

            quan_total_flops, quan_total_params= print_model_parm_flops(flops,params,can[:-2])
            if  quan_total_params/total_params >= args.avg_bw_weights:
                continue
            can[-1] = quan_total_flops
            can[-2] = quan_total_params

            # ## Discrete recombination
            # mask = np.random.randint(low=0, high=2, size=(num_states+2)).astype(np.float32)
            # can = p1*mask + p2*(1.0-mask)
            iter += 1
            # print(total_flops/sparse_total_flops)
            # print(total_params/sparse_total_params)

            # compare difference
            t_can_int = [i for i in t_can]
            if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                continue
            else:
                candidates_int.append(t_can_int)

            if (can.tolist() not in res) and (can.tolist() not in keep_top_k):
                res.append(can.tolist())
                if len(res)==crossover_num:
                    break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res

# random select operation in evolution algorithm
def bw_fm_random_can(args,num, num_states, featuremaps, total_featuremaps):
    print('random select ........', flush=True)
    candidates = []
    while(len(candidates))<num:
        if args.fix_bw_fm == True:
            can = np.ones(num_states+2)*args.avg_bw_fm
            can[-1] = total_featuremaps*args.avg_bw_fm
            can[-2] = total_featuremaps*args.avg_bw_fm
        else:
            # # uniform random init
            # can = np.random.randint(1,np.ceil(args.avg_bw_fm*2),num_states+2) #1.5
            can = np.random.randint(args.min_bw,args.max_bw,num_states+2) #1.5
            can[0] = np.ceil(args.avg_bw_fm*3) if np.ceil(args.avg_bw_fm*3) < args.max_bw else args.max_bw 
            can[-3] = np.ceil(args.avg_bw_fm*3) if np.ceil(args.avg_bw_fm*3) < args.max_bw else args.max_bw 
            # can = np.random.randint(1,args.max_bw,num_states+2)
            # normal_init
            # can =  np.round(np.random.normal(loc=0,scale=1,size=num_states))

            mask = can[:-2] > args.max_bw
            can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
            mask = can[:-2] < args.min_bw
            can[:-2] = can[:-2]*(1-mask)+args.min_bw*mask

            quan_total_featuremaps = print_model_featuremap(featuremaps,can[:-2])
            if  quan_total_featuremaps/total_featuremaps >= args.avg_bw_fm:
                continue
            can[-1] = quan_total_featuremaps
            can[-2] = quan_total_featuremaps

        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates

# mutation operation in evolution algorithm
def bw_fm_get_mutation(args,epoch, keep_top_k, top_k_acc,num_states, mutation_num, m_prob,strength,max_range, featuremaps, total_featuremaps):
    print('mutation ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for g_can in keep_top_k:
        global_candidates_int.append([i for i in g_can[:-2]])
    k = len(keep_top_k)
    iter = 0
    max_iters = 10*mutation_num
    # top_k_acc = top_k_acc / 100.0
    top_k_p = bw_softmax_numpy(top_k_acc)
    # while len(res)<mutation_num and iter<max_iters:
    while len(res)<mutation_num:
        if args.fix_bw_fm == True:
            can = np.ones(num_states+2)*args.avg_bw_fm
            can[-1] = total_featuremaps*args.avg_bw_fm
            can[-2] = total_featuremaps*args.avg_bw_fm
            res.append(can.tolist())
        else:
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
            alpha = alpha*(1-mask1)+0.5*(mask1) 
            alpha = alpha*(1-mask2)+0.5*(mask2)
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

                mask = can[:-2] > args.max_bw
                can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
                mask = can[:-2] < args.min_bw
                can[:-2] = can[:-2]*(1-mask)+args.min_bw*mask

                t_can = tuple(can[:-2])

                quan_total_featuremaps = print_model_featuremap(featuremaps,can[:-2])
                if  quan_total_featuremaps/total_featuremaps >= args.avg_bw_fm:
                    continue
                can[-1] = quan_total_featuremaps
                can[-2] = quan_total_featuremaps

                # compare difference
                t_can_int = [i for i in t_can]
                if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                    continue
                else:
                    candidates_int.append(t_can_int)

                if (can.tolist() not in res) and (can.tolist() not in keep_top_k):
                    cnt = cnt + 1
                    res.append(can.tolist())
                    if len(res)==mutation_num:
                        break
    print('mutation_num = {}'.format(len(res)), flush=True)
    return res

# crossover operation in evolution algorithm
def bw_fm_get_crossover(args,keep_top_k, top_k_acc, num_states, crossover_num, featuremaps, total_featuremaps):
    print('crossover ......', flush=True)
    res = []
    candidates_int = []
    global_candidates_int = []
    for g_can in keep_top_k:
        global_candidates_int.append([i for i in g_can[:-2]])
    k = len(keep_top_k)
    # top_k_acc = top_k_acc / 100.0
    top_k_p = bw_softmax_numpy(top_k_acc)
    iter = 0
    max_iters = 10 * crossover_num
    while len(res)<crossover_num:
        if args.fix_bw_fm == True:
            can = np.ones(num_states+2)*args.avg_bw_fm
            can[-1] = total_featuremaps*args.avg_bw_fm
            can[-2] = total_featuremaps*args.avg_bw_fm
            res.append(can.tolist())
        else:
            # id1, id2 = np.random.choice(k, 2, replace=False,p=top_k_p)
            # id1, id2 = np.random.choice(k, 2, replace=False)
            # print([id1,id2])
            # p1 = keep_top_k[id1]
            # p2 = keep_top_k[id2]
            id1 = np.random.choice(k, 1, p=top_k_p)   
            id2 = np.random.choice(k, 1)
            # print([id1[0],id2[0]])
            p1 = keep_top_k[id1[0]]
            p2 = keep_top_k[id2[0]]
            
            ## recombination

            alpha = np.random.rand(len(p1))
            p1 = np.array(p1)
            p2 = np.array(p2)
            can = np.round(p1*alpha + p2*(1.0-alpha))

            # alpha = np.random.uniform(-1, 1, len(p1))
            # p1 = np.array(p1)
            # p2 = np.array(p2)
            # can = np.round(p1+alpha*(p1-p2))

            m = can <= 0
            can = can*(1.0-m)+m

            mask = can[:-2] > args.max_bw
            can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask
            mask = can[:-2] < args.min_bw
            can[:-2] = can[:-2]*(1-mask)+args.min_bw*mask

            t_can = tuple(can[:-2])

            quan_total_featuremaps = print_model_featuremap(featuremaps,can[:-2])
            if  quan_total_featuremaps/total_featuremaps >= args.avg_bw_fm:
                continue
            can[-1] = quan_total_featuremaps
            can[-2] = quan_total_featuremaps

            # ## Discrete recombination
            # mask = np.random.randint(low=0, high=2, size=(num_states+2)).astype(np.float32)
            # can = p1*mask + p2*(1.0-mask)
            iter += 1
            # print(total_flops/sparse_total_flops)
            # print(total_params/sparse_total_params)

            # compare difference
            t_can_int = [i for i in t_can]
            if (t_can_int in candidates_int) or (t_can_int in global_candidates_int):
                continue
            else:
                candidates_int.append(t_can_int)

            if (can.tolist() not in res) and (can.tolist() not in keep_top_k):
                res.append(can.tolist())
                if len(res)==crossover_num:
                    break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res
