import os
import numpy as np
import sys

def bw_softmax_numpy(x):
    e_x = np.exp(x.astype(float) - np.max(x))
    return e_x / e_x.sum()

def bw_jugdment_value(args, bw):
    mask = bw <= 0
    bw = bw * (1.0 - mask) + mask
    mask = bw > args.max_bw
    bw = bw * (1 - mask) + args.max_bw * mask
    bw = np.round(bw)
    return bw

def bw_print_model_parm_flops(flops, params, bw):
    quan_flops = [a * b for a, b in zip(bw, flops)]
    quan_params = [a * b for a, b in zip(bw, params)]
    quan_total_flops = sum(quan_flops)
    quan_total_params = sum(quan_params)
    return quan_total_flops,quan_total_params

def bw_random_can(args, num, num_states, flops, params, total_flops, total_params):
    print('randomly selecting for quantization........', flush=True)
    candidates = []
    while(len(candidates))<num:
        # # uniform random init
        can = np.random.randint(1, np.ceil(args.avg_bw * 2) + 3, num_states + 3) #1.5
        can[0] = 0
        # normal_init
        # can =  np.round(np.random.normal(loc=0,scale=1,size=num_states))

        mask = can[1:-2] > args.max_bw
        can[1:-2] = can[1:-2] * (1 - mask) + args.max_bw * mask

        quan_total_flops, quan_total_params = bw_print_model_parm_flops(flops, params, can[1:-2])
        if  quan_total_params / total_params >= args.avg_bw:
            continue
        can[-1] = quan_total_flops
        can[-2] = quan_total_params

        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return np.array(candidates)

def bw_get_mutation(args, top_k_bees, top_k_acc, num_states, strength, max_range, flops, params, total_flops, total_params):
    print('mutation for quantization......', flush=True)
    child = np.empty([len(top_k_acc), num_states + 3])
    for i, bee in enumerate(top_k_bees):
        # mutate
        while True:
            iters = 0
            max_iters = 2 * num_states
            while True and iters < max_iters:
                iters += 1
                alpha = np.random.normal(0.5, 0.1, len(bee) - 1)
                mask = alpha < 0.5
                beta = (mask * ( (2*alpha)**(1 / strength) - 1 ) + (1 - mask) * (1 - (2 - 2 * alpha)**(1 / strength))) * max_range
                beta = beta * 4
                can_bw = np.copy(bee[1:])
                can_bw = np.round(beta + can_bw)

                mask = can_bw <= 0
                can_bw = mask + (1 - mask) * can_bw
                mask = can_bw > args.max_bw
                can_bw = mask * args.max_bw + (1 - mask) * can_bw
                
                if (can_bw[:-2].tolist() in top_k_bees[:, 1:-2].tolist()) or (can_bw[:-2].tolist() in child[:, 1:-2].tolist()):
                    continue
                break
            quan_total_flops, quan_total_params = bw_print_model_parm_flops(flops, params, can_bw[:-2])
            if  quan_total_params/total_params >= args.avg_bw:
                continue
            can_bw[-1] = quan_total_flops
            can_bw[-2] = quan_total_params
            break
        child[i, 0] = np.copy(bee[0])
        child[i, 1:] = can_bw

        # # empolyed bees
        # while True:
        #     while True:
        #         cross_id = np.random.randint(low=0, high=len(top_k_acc))
                
        #         can_bw1 = np.copy(bee[1:])
        #         can_bw2 = np.copy(top_k_bees[cross_id][1:])

        #         r = np.random.uniform(-1, 1, len(bee) - 1)
        #         can_bw1 = (1 + r) * can_bw1 - r * can_bw2

        #         can_bw1 = np.round(can_bw1)

        #         mask = can_bw1 <= 0
        #         can_bw1 = can_bw1 * (1.0 - mask) + mask
        #         mask = can_bw1 > args.max_bw
        #         can_bw1 = can_bw1 * (1 - mask) + args.max_bw * mask

        #         if (can_bw1[:-2].tolist() in top_k_bees[:, 1:-2].tolist()) or (can_bw1[:-2].tolist() in child[:, 1:-2].tolist()):
        #             continue
        #         break
        #     quan_total_flops, quan_total_params = bw_print_model_parm_flops(flops, params, can_bw1[:-2])
        #     if  quan_total_params / total_params >= args.avg_bw:
        #         continue
        #     can_bw1[-1] = quan_total_flops
        #     can_bw1[-2] = quan_total_params
        #     break
        # child[i, 0] = np.copy(bee[0])
        # child[i, 1:] = can_bw1

    print("mutation finished", flush=True)
    print('mutation_num = {}\n'.format(len(top_k_acc)), flush=True)
    return child

def bw_get_crossover(args, top_k_bees, top_k_acc, num_states, flops, params, total_flops, total_params):
    print('crossover for quantization......', flush=True)
    k = len(top_k_bees)
    child = np.empty([k, num_states + 3])
    select_id = []
    for i, bee in enumerate(top_k_bees):
        # onlookers
        top_k_p = bw_softmax_numpy(top_k_acc)
        while True:
            iters = 0
            max_iter = 2 * num_states
            while True and iters < max_iter:
                iters += 1
                cross_id1 = np.random.choice(range(k), p=top_k_p)
                # cross_id1 = np.random.randint(low=0, high=k)
                cross_id2 = np.random.randint(low=0, high=k)
                
                can_bw1 = np.copy(top_k_bees[cross_id1][1:])
                can_bw2 = np.copy(top_k_bees[cross_id2][1:])

                # r = np.random.uniform(-1, 1)
                # cross_dim = np.random.randint(low=0, high=int(num_states))
                # can_bw2[cross_dim] = (1 + r) * can_bw1[cross_dim] - r * can_bw2[cross_dim]

                r = np.random.uniform(0, 1, len(bee) - 1)
                can_bw1 = r * can_bw1 + (1 - r) * can_bw2

                can_bw1 = np.round(can_bw1)

                mask = can_bw1 <= 0
                can_bw1 = can_bw1 * (1.0 - mask) + mask
                mask = can_bw1 > args.max_bw
                can_bw1 = can_bw1 * (1 - mask) + args.max_bw * mask

                if (can_bw1[:-2].tolist() in child[:, 1:-2].tolist()) or (can_bw1[:-2].tolist() in top_k_bees[:, 1:-2].tolist()):
                    continue
                break
            quan_total_flops, quan_total_params = bw_print_model_parm_flops(flops, params, can_bw1[:-2])
            if  quan_total_params / total_params >= args.avg_bw:
                continue
            can_bw1[-1] = quan_total_flops
            can_bw1[-2] = quan_total_params
            break
        select_id.append(cross_id1)
        child[i, 0] = np.copy(top_k_bees[cross_id1][0])
        child[i, 1:] = can_bw1
    print("crossover finished")
    print('crossover_num = {}\n'.format(k), flush=True)
    return child, select_id
