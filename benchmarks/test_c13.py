import os, sys
from pathlib import Path
proj_path = str(Path(__file__).resolve().parents[1])
sys.path.append(proj_path)
import random
import timeit
from copy import deepcopy
import csv
import pandas as pd
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


def train_barrier(example_name, controller,b1_hidden_neurons,batch_size,lr,R_b):
    b1_hidden_neurons=[b1_hidden_neurons]

    b1_activations = ['SKIP']
    b1_hidden_neurons = b1_hidden_neurons * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)


    start = timeit.default_timer()
    opts = {
        'b_act': b1_activations,
        'b_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': [],
        'bm1_hidden': [],
        "batch_size": batch_size,
        'lr': lr,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': R_b,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        'max_iter': 10
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    vis, barrier, t,loss,iterbc = cegis.solve()
    return vis, barrier, t,loss,iterbc


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    ###

    b1_hidden_neurons_values = [15]
    
    batch_sizes = [1000, 1500, 2000, 5000]
    learning_rates = [0.001, 0.01,0.02,0.03,0.04, 0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    R_b_values = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


    csv_file = 'C13_hyperparameter_tuning_results.csv'


    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iter', 'iterbc', 'b1_hidden_neurons', 'batch_size', 'lr', 'R_b',
                         'all_train_time', 'all_bc_learn_time', 'all_counter_example_time', 'all_verify_time'])

    example_name = 'C13'
    iter = 2
    agent, env, replay_buffer = ddpg_init(example_name)


    for b1_hidden_neurons in b1_hidden_neurons_values:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for R_b in R_b_values:
                    all_train_time = 0
                    all_bc_learn_time = 0
                    all_counter_example_time = 0
                    all_verify_time = 0

                    for i in range(iter):

                        tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
                        all_train_time += train_time


                        controller = [tmp_controller]
                        print(controller)
                        vis, barrier, t, loss, iterbc = train_barrier(
                            example_name, controller, b1_hidden_neurons, batch_size, lr, R_b
                        )

                        all_bc_learn_time += t[0]
                        all_counter_example_time += t[1]
                        all_verify_time += t[2]

                        if vis:
                            print(f"Suc_Controller iteration: {i + 1}")
                            print(f"Suc_All training time: {all_train_time}")
                            print(f"Suc_All barrier learning time: {all_bc_learn_time}")
                            print(f"Suc_All counter-example generation time: {all_counter_example_time}")
                            print(f"Suc_All verification time: {all_verify_time}")
                            break

                        print(f"Controller iteration: {i + 1}")
                        print(f"All training time: {all_train_time}")
                        print(f"All barrier learning time: {all_bc_learn_time}")
                        print(f"All counter-example generation time: {all_counter_example_time}")
                        print(f"All verification time: {all_verify_time}")


                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([i + 1, iterbc, b1_hidden_neurons, batch_size, lr, R_b,
                                         all_train_time, all_bc_learn_time,
                                         all_counter_example_time, all_verify_time,controller,barrier])

                    print(f"hidden_neurons: {b1_hidden_neurons}, batch_size: {batch_size}, lr: {lr}, R_b: {R_b}, controller: {controller}, barrier {barrier}")
                    print("-----------------------------------------")


