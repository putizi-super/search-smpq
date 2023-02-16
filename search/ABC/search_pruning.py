import numpy as np
import sys


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

def judgment_value(args, a, min, max):
    mask = a < min
    a = (0.5 / args.max_PARAMs) * mask + a * (1 - mask)
    # a = min * mask + a * (1 - mask)
    mask = a > max
    a = max * mask + a * (1 - mask)
    return a

def random_can(args, num, num_states, flops, params, total_flops, total_params):
    print('randomly selecting for pruning........', flush=True)
    candidates = []
    while(len(candidates)) < num:
        # # uniform random init
        # high = 2.0/max(args.max_PARAMs,args.max_FLOPs)
        # can = np.random.uniform(0.001, high, num_states+2)

        # # normal init
        # loc = 1/max(args.max_PARAMs,args.max_FLOPs)
        # scale = (0.6-loc)/3.0
        # can = uniform_normal_init(loc,scale,num_states+2)

        # # mix normal_init
        # loc = 1.0 / max(args.max_PARAMs, args.max_FLOPs)
        # can = mix_normal_init(loc, num_states + 2, 0.99)

        # gamma init
        loc = 1.0 / max(args.max_PARAMs, args.max_FLOPs)
        r = np.random.uniform(10, 64)
        can = np.random.gamma(loc * r, 1./r, num_states + 3)

        mask = can > 0.99
        can = mask * 0.99 + (1 - mask) * can
        can[0] = 0

        sparse_total_flops, sparse_total_params = print_model_parm_flops(total_flops, total_params, flops, params, can[1:-2])
        if (total_flops / sparse_total_flops < args.max_FLOPs) or (total_params / sparse_total_params < args.max_PARAMs):
            continue
        can[-1] = sparse_total_flops
        can[-2] = sparse_total_params
        candidates.append(can)
    print('random_num = {}'.format(len(candidates)), flush=True)
    return np.array(candidates)

def get_mutation(args, top_k_bees, top_k_acc, num_states, strength, flops, params, total_flops, total_params):
    print('mutation for pruning......', flush=True)
    child = np.empty([len(top_k_acc), num_states + 3])
    for i, bee in enumerate(top_k_bees):
        # mutate
        while True:
            while True :
                alpha = np.random.normal(0.5, 0.1, len(bee) - 1)
                mask = alpha < 0.5
                beta = mask * ( (2*alpha)**(1/strength) - 1 ) + (1 - mask) * ( 1 - (2 - 2 * alpha)**(1/strength) )
                can_p = np.copy(bee[1:])
                can_p = beta + can_p

                can_p = judgment_value(args, can_p, 0.001, 0.99)

                if (can_p[:-2].tolist() in top_k_bees[:, 1:-2].tolist()) or (can_p[:-2].tolist() in child[:, 1:-2].tolist()):
                    continue
                break
            sparse_total_flops, sparse_total_params = print_model_parm_flops(total_flops, total_params, flops, params, can_p[:-2])
            if (total_flops / sparse_total_flops < args.max_FLOPs) or (total_params / sparse_total_params < args.max_PARAMs):
                continue
            can_p[-1] = sparse_total_flops
            can_p[-2] = sparse_total_params
            break
        child[i, 0] = np.copy(bee[0])
        child[i, 1:] = can_p

        # # empolyed bees
        # while True:
        #     while True :
        #         cross_id = np.random.randint(low=0, high=len(top_k_acc))
                
        #         can_p1 = np.copy(bee[1:])
        #         can_p2 = np.copy(top_k_bees[cross_id][1:])

        #         r = np.random.uniform(-1, 1, len(bee) - 1)
        #         can_p1 = (1 + r) * can_p1 - r * can_p2

        #         can_p1 = judgment_value(args, can_p1, 0.001, 0.99)

        #         if (can_p1[:-2].tolist() in top_k_bees[:, 1:-2].tolist()) or (can_p1[:-2].tolist() in child[:, 1:-2].tolist()):
        #             continue
        #         break
        #     sparse_total_flops, sparse_total_params = print_model_parm_flops(flops, params, can_p1[:-2])
        #     if (total_flops / sparse_total_flops < args.max_FLOPs) or (total_params / sparse_total_params < args.max_PARAMs):
        #         continue
        #     can_p1[-1] = sparse_total_flops
        #     can_p1[-2] = sparse_total_params
        #     break
        # child[i, 0] = np.copy(bee[0])
        # child[i, 1:] = can_p1

    print("mutation finished", flush=True)
    print('mutation_num = {}\n'.format(len(top_k_acc)), flush=True)
    return child

def get_crossover(args, top_k_bees, top_k_acc, num_states, flops, params, total_flops, total_params):
    print('crossover for pruning......', flush=True)
    k = len(top_k_bees)
    child = np.empty([k, num_states + 3])
    select_id = []
    for i, bee in enumerate(top_k_bees):
        top_k_p = softmax_numpy(top_k_acc)
        while True:
            while True:
                cross_id1 = np.random.choice(range(k), p=top_k_p)
                # cross_id1 = np.random.randint(low=0, high=k)
                cross_id2 = np.random.randint(low=0, high=k)

                can_p1 = np.copy(top_k_bees[cross_id1][1:])
                can_p2 = np.copy(top_k_bees[cross_id2][1:])

                # r = np.random.uniform(-1, 1)
                # cross_dim = np.random.randint(low=0, high=int(num_states))
                # can_p2[cross_dim] = (1 + r) * can_p1[cross_dim] - r * can_p2[cross_dim]

                r = np.random.uniform(0, 1, len(bee) - 1)
                can_p1 = r * can_p1 + (1 - r) * (can_p2)

                can_p1 = judgment_value(args, can_p1, 0.001, 0.99)

                if (can_p1[:-2].tolist() in top_k_bees[:, 1:-2].tolist()) or (can_p1[:-2].tolist() in child[:, 1:-2].tolist()):
                    continue
                break
            sparse_total_flops, sparse_total_params = print_model_parm_flops(total_flops, total_params, flops, params, can_p1[:-2])
            if (total_flops / sparse_total_flops < args.max_FLOPs) or (total_params / sparse_total_params < args.max_PARAMs):
                continue
            can_p1[-1] = sparse_total_flops
            can_p1[-2] = sparse_total_params
            break
        select_id.append(cross_id1)
        child[i, 0] = np.copy(top_k_bees[cross_id1][0])
        child[i, 1:] = can_p1
    print("crossover finished")
    print('crossover_num = {}\n'.format(k), flush=True)
    return child, select_id
