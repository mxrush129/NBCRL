import sys, os
import random
import timeit
from copy import deepcopy
import pickle
import torch
import numpy as np
import sympy as sp
from utils.Config import CegisConfig
from Examples import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis
from RL.train_controller import train_by_ddpg, ddpg_init

def save_object(obj, filename):
    if not os.path.exists(filename):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Object saved to {filename}")

def load_object(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object loaded from {filename}")
        return obj
    else:
        print(f"No saved object found at {filename}")
        return None
    
    
def update_f(example, u):
    x = sp.symbols([f'x{i + 1}' for i in range(example.n_obs)])
    f = [ff(x, u) for ff in example.f]
    return [sp.lambdify(x, ff) for ff in f]


def train_barrier(example_name, controller):
    b1_activations = ['SKIP']
    b1_hidden_neurons = [10] * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)

    start = timeit.default_timer()
    opts = {
        'b_act': b1_activations,
        'b_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': ['LINEAR'],
        'bm1_hidden': [5],
        "batch_size": 500,
        'lr': 0.1,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.6,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        'max_iter': 5
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    vis, barrier, t, current_loss= cegis.solve()
    return vis, barrier, t,current_loss


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    example_name = 'C2'
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
        
        vis, barrier, t,current_loss = train_barrier(example_name, controller)
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

