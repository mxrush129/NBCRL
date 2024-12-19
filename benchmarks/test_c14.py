import os, sys
import sys, os
import sys

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
        # todo
        'max_iter': 25
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    # print(cegis.solve())
    vis, barrier, t,loss,iterbc = cegis.solve()
    return vis, barrier, t,loss,iterbc


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    b1_hidden_neurons_values = [20]
    batch_sizes = [5000]
    learning_rates = [0.001, 0.01,0.02,0.03,0.04, 0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    R_b_values = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


    csv_file = '5000skip_C14_hyperparameter_tuning_results.csv'


    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iter', 'iterbc', 'b1_hidden_neurons', 'batch_size', 'lr', 'R_b',
                         'all_train_time', 'all_bc_learn_time', 'all_counter_example_time', 'all_verify_time'])

    example_name = 'C14'
    iter = 1
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

                        # tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
                        # all_train_time += train_time


                        # controller = [tmp_controller]
                        x1, x2, x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
                        controller=[-45.74866703453*x1**2 + 1.4318941533116*x1*x10 + 0.697919669384808*x1*x11 - 11.6602517525944*x1*x12 - 6.03952533416076*x1*x2 - 2.14030994316214*x1*x3 + 9.45489566091378*x1*x4 + 0.155242768528543*x1*x5 + 7.04865379107685*x1*x6 - 1.37133795891202*x1*x7 - 2.91321849267269*x1*x8 - 11.3557438592015*x1*x9 + 0.385261882485526*x1 - 13.4941154410221*x10**2 - 2.45980870755393*x10*x11 + 28.3960455553006*x10*x12 - 7.0288575754053*x10*x2 - 10.6899678442378*x10*x3 - 1.03672166646185*x10*x4 + 2.36819577888762*x10*x5 - 4.39037351747808*x10*x6 + 8.19981952779022*x10*x7 - 2.35321976630488*x10*x8 + 2.34923626885989*x10*x9 + 2.1844289616366*x10 - 10.7639791251429*x11**2 + 21.6053508321644*x11*x12 - 1.51345578141036*x11*x2 - 1.69998452466909*x11*x3 - 1.71742949317999*x11*x4 - 0.295002742440834*x11*x5 + 3.08998740749518*x11*x6 + 1.37011122495887*x11*x7 + 8.68555292221082*x11*x8 - 3.4604423951823*x11*x9 + 0.995150704708196*x11 - 51.3776751974758*x12**2 + 10.2375311059926*x12*x2 + 8.26983571306904*x12*x3 + 1.09154470118874*x12*x4 - 2.85733831289763*x12*x5 + 1.31083466398343*x12*x6 - 5.41661962299567*x12*x7 - 1.80687705635448*x12*x8 + 16.8860649337644*x12*x9 + 4.95227367768634*x12 - 45.5233283358111*x2**2 + 3.16078009647757*x2*x3 - 4.46553194998215*x2*x4 + 12.820293056699*x2*x5 + 14.2188677016897*x2*x6 - 2.77973511157536*x2*x7 - 5.76030294580671*x2*x8 + 5.37414057235245*x2*x9 + 0.0234075049638443*x2 - 31.0277518667281*x3**2 - 4.3456924241939*x3*x4 - 1.34740190850868*x3*x5 - 14.6502718782418*x3*x6 - 23.2165332614596*x3*x7 + 6.66822956972218*x3*x8 - 10.3237417722594*x3*x9 - 0.0249801432798992*x3 + 0.873115128795324*x4**2 - 0.230801840577822*x4*x5 - 4.88580206208692*x4*x6 - 1.4124057563269*x4*x7 - 0.856194889766453*x4*x8 - 0.350633922358373*x4*x9 + 0.38327226496664*x4 + 1.25961741695883*x5**2 + 0.394643216361699*x5*x6 + 3.91125055512187*x5*x7 + 0.019254811236396*x5*x8 + 1.18175443690613*x5*x9 - 0.719903651964604*x5 - 26.2584606187256*x6**2 - 6.68267824704954*x6*x7 + 10.1354664363125*x6*x8 - 8.96996441745714*x6*x9 - 0.341561117588179*x6 - 24.2548195588089*x7**2 - 6.02699148915257*x7*x8 + 7.63519867926001*x7*x9 - 0.79332631895939*x7 - 30.5242713819171*x8**2 + 3.71130232983274*x8*x9 - 0.0122329286938562*x8 - 21.7764196782607*x9**2 + 2.02981899995023*x9]

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

