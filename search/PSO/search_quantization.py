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

def judgment_value(args,a):
    mask = a > args.max_bw
    a = a*(1-mask)+args.max_bw*mask

    mask = a < 1.0
    a = a*(1-mask)+1.0*mask
    return a

# random select operation in evolution algorithm
def bw_random_can(args,num, num_states,flops, params, total_flops, total_params):
    print('random select ........', flush=True)
    x = []
    v = []
    while(len(x))<num:
        # # uniform random init
        can = np.random.randint(1,np.ceil(args.avg_bw*2)+3,num_states+2) #1.5
        # normal_init
        # can =  np.round(np.random.normal(loc=0,scale=1,size=num_states))

        mask = can[:-2] > args.max_bw
        can[:-2] = can[:-2]*(1-mask)+args.max_bw*mask

        quan_total_flops, quan_total_params= print_model_parm_flops(flops,params,can[:-2])
        if  quan_total_params/total_params >= args.avg_bw:
            continue
        can[-1] = quan_total_flops
        can[-2] = quan_total_params

        x.append(can)
        v.append(np.zeros(len(can)))
    print('random_num = {}'.format(len(x)), flush=True)
    print('random_num = {}'.format(len(v)), flush=True)
    return  x, v

 
# crossover operation in evolution algorithm
def bw_get_crossover(args, v_bw, x_bw, p_best, g_best, flops, params, total_flops, total_params):
    print('crossover ......', flush=True)

    for i,bw in enumerate(x_bw):
        while True:
            alpha1 = np.random.rand(len(g_best[0]))
            alpha2 = np.random.rand(len(g_best[0]))
            v_bw[i] = 0.8*v_bw[i] + 0.6*alpha1*(p_best[i][0]-bw) + 0.3*alpha2*(g_best[0]-bw)
            x_bw[i] = np.round(bw + v_bw[i])
            x_bw[i] = judgment_value(args,x_bw[i])
            quan_total_flops, quan_total_params= print_model_parm_flops(flops,params,x_bw[i][:-2])
            if  quan_total_params/total_params >= args.avg_bw:
                continue
            x_bw[i][-1] = quan_total_flops
            x_bw[i][-2] = quan_total_params  
            break          

    return x_bw,v_bw