import random
import timeit
from copy import deepcopy


import torch
import numpy as np
import sympy as sp
from utils.Config import CegisConfig
from Examples import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis
from RL.train_controller import train_by_ddpg, ddpg_init


def update_f(example, u):
    x = sp.symbols([f'x{i + 1}' for i in range(example.n_obs)])
    f = [ff(x, u) for ff in example.f]
    return [sp.lambdify(x, ff) for ff in f]


def train_barrier(example_name, controller):
    b1_activations = ['SKIP']
    b1_hidden_neurons = [20] * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)

    start = timeit.default_timer()
    opts = {
        'b_act': b1_activations,
        'b_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': ['LINEAR'],
        'bm1_hidden': [5],
        "batch_size": 150,
        'lr': 0.05,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.6,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        'max_iter': 5
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    
    vis, barrier, t,loss = cegis.solve()
    return vis, barrier, t


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    example_name = 'C4'
    # todo
    iter = 10
    B = None
    all_train_time = 0
    all_bc_learn_time = 0
    all_counter_example_time = 0
    all_verify_time = 0

    agent, env, replay_buffer = ddpg_init(example_name)
    for i in range(iter):
        tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
        all_train_time += train_time
        controller = [tmp_controller]
        print(controller)
        x1, x2 = sp.symbols('x1 x2')
        controller=[-0.122333688556166*x1**2 - 0.481137437777661*x1*x2 + 1.43916680467694*x1 - 0.629216073444153*x2**2 - 0.469764004097184*x2]
        print(controller)
        
        vis, barrier, t = train_barrier(example_name, controller)
        all_bc_learn_time += t[0]
        all_counter_example_time += t[1]
        all_verify_time += t[2]

        if vis:
            print(f"controller iteration: {i + 1}")
            print(f"all iteration ddpg train time: {all_train_time}")
            print(f"all iteration learn time: {all_bc_learn_time}")
            print(f"all iteration counter examples generate time: {all_counter_example_time}")
            print(f"all iteration sos verify time: {all_verify_time}")

            break
        B = barrier
        env.update_barrier(B)

