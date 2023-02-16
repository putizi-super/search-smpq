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
    a = (0.5/args.max_PARAMs)*m1+a*(1-m1)
    m2 = a > max
    a = max*m2+a*(1-m2)
    return a

# random select operation in evolution algorithm
def random_can(args, num, num_states, flops, params, total_flops, total_params):
    print('random select ........', flush=True)
    x = []
    v = []
    while(len(x))<num:
        # # uniform random init
        # high = 2.0/max(args.max_PARAMs,args.max_FLOPs)
        # can = np.random.uniform(0.001,high,num_states+2)

        # ## uniform normal init
        # loc = 1/max(args.max_PARAMs,args.max_FLOPs)
        # scale = (0.6-loc)/3.0
        # can = uniform_normal_init(loc,scale,num_states+2)

        # # mix normal_init
        loc = 1.0/max(args.max_PARAMs,args.max_FLOPs)
        can = mix_normal_init(loc,num_states+2,0.99)


        t_can = tuple(can[:-2])
        sparse_total_flops, sparse_total_params= print_model_parm_flops(total_flops,total_params,flops,params,can[:-2])
        if (total_flops/sparse_total_flops < args.max_FLOPs) or (total_params/sparse_total_params < args.max_PARAMs):
            continue
        can[-1] = sparse_total_flops
        can[-2] = sparse_total_params
        x.append(can)
        v.append(np.zeros(len(can)))
    print('random_num = {}'.format(len(x)), flush=True)
    print('random_num = {}'.format(len(v)), flush=True)
    return x, v

# crossover operation in PSO
def get_crossover(args,v_prune,x_prune,p_best,g_best,flops,params,total_flops,total_params):
    print('crossover ......', flush=True)

    for i,prune in enumerate(x_prune):
        while True:
            alpha1 = np.random.rand(len(g_best[1]))
            alpha2 = np.random.rand(len(g_best[1]))
            # print(p_best[i][1])
            # print(g_best[1])
            v_prune[i] = 0.8*v_prune[i] + 0.6*alpha1*(p_best[i][1]-prune) + 0.3*alpha2*(g_best[1]-prune)
            x_prune[i] = prune + v_prune[i]
            # print(v_prune[i])
            # print(x_prune[i])

            # x_prune[i] = judgment_value(args,x_prune[i],0.001, 1.5/max(args.max_PARAMs,args.max_FLOPs))
            x_prune[i] = judgment_value(args,x_prune[i],0.001, 0.999)

            # print(x_prune[i])

            # sys.exit()


            sparse_total_flops, sparse_total_params= print_model_parm_flops(total_flops,total_params,flops,params,x_prune[i][:-2])
            if (total_flops/sparse_total_flops < args.max_FLOPs) or (total_params/sparse_total_params < args.max_PARAMs):
                continue

            # print(i)

            x_prune[i][-1] = sparse_total_flops
            x_prune[i][-2] = sparse_total_params  
            break          

    return x_prune,v_prune
