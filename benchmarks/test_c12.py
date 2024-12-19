import os, sys
import sys, os
import sys
sys.path.append('/Users/hary/Desktop/投稿文章/DAC24_TCAD投稿/codes/TCAD')
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

    b1_activations = ['MUL']
    b1_hidden_neurons = b1_hidden_neurons * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)


    start = timeit.default_timer()
    opts = {
        'b_act': b1_activations,
        'b_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': ["LINEAR"],
        'bm1_hidden': [5],
        "batch_size": batch_size,
        'lr': lr,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': R_b,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        # todo
        'max_iter': 5
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
    ###

    b1_hidden_neurons_values = [20]
    batch_sizes = [250,300,350,400,500, 1000, 1500, 2000, 5000]
    learning_rates = [0.001, 0.01, 0.05,0.1,0.2,0.3,0.4,0.5]
    R_b_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # 结果文件名
    csv_file = 'hyperparameter_tuning_results.csv'

    # 写入CSV文件头
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iter', 'iterbc', 'b1_hidden_neurons', 'batch_size', 'lr', 'R_b',
                         'all_train_time', 'all_bc_learn_time', 'all_counter_example_time', 'all_verify_time'])

    example_name = 'C12'
    iter = 2  # 最大迭代次数
    agent, env, replay_buffer = ddpg_init(example_name)

    # 网格搜索参数组合
    for b1_hidden_neurons in b1_hidden_neurons_values:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for R_b in R_b_values:
                    all_train_time = 0
                    all_bc_learn_time = 0
                    all_counter_example_time = 0
                    all_verify_time = 0

                    for i in range(iter):
                        # 训练控制器
                        tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
                        all_train_time += train_time

                        # 设置控制器并训练 barrier
                        controller = [tmp_controller]
                        print(controller)
                        vis, barrier, t, loss, iterbc = train_barrier(
                            example_name, controller, b1_hidden_neurons, batch_size, lr, R_b
                        )

                        all_bc_learn_time += t[0]
                        all_counter_example_time += t[1]
                        all_verify_time += t[2]

                        # if vis:
                        #     print(f"Controller iteration: {i + 1}")
                        #     print(f"All training time: {all_train_time}")
                        #     print(f"All barrier learning time: {all_bc_learn_time}")
                        #     print(f"All counter-example generation time: {all_counter_example_time}")
                        #     print(f"All verification time: {all_verify_time}")
                        #     break

                        print(f"Controller iteration: {i + 1}")
                        print(f"All training time: {all_train_time}")
                        print(f"All barrier learning time: {all_bc_learn_time}")
                        print(f"All counter-example generation time: {all_counter_example_time}")
                        print(f"All verification time: {all_verify_time}")

                    # 保存结果到CSV
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([i + 1, iterbc, b1_hidden_neurons, batch_size, lr, R_b,
                                         all_train_time, all_bc_learn_time,
                                         all_counter_example_time, all_verify_time,controller])

                    print(f"hidden_neurons: {b1_hidden_neurons}, batch_size: {batch_size}, lr: {lr}, R_b: {R_b}, controller:{controller}")
                    print("-----------------------------------------")

    # results = []
    # ###
    # example_name = 'C12'
    # # todo
    # iter = 2
    # B = None
    # all_train_time = 0
    # all_bc_learn_time = 0
    # all_counter_example_time = 0
    # all_verify_time = 0
    # # 需要一个example_name
    # agent, env, replay_buffer = ddpg_init(example_name)
    #
    # for b1_hidden_neurons in b1_hidden_neurons_values:
    #     for batch_size in batch_sizes:
    #         for lr in learning_rates:
    #             for R_b in R_b_values:
    #                 # 初始化累计时间
    #                 all_train_time = 0
    #                 all_bc_learn_time = 0
    #                 all_counter_example_time = 0
    #                 all_verify_time = 0
    #
    #                 # 训练和验证过程
    #                 for i in range(iter):
    #                     # 训练控制器
    #                     tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
    #                     all_train_time += train_time
    #
    #                     # 设置控制器
    #                     controller = [tmp_controller]
    #                     print(controller)
    #                     # 训练 barrier
    #                     vis, barrier, t,loss,iterbc = train_barrier(example_name, controller,b1_hidden_neurons,batch_size,lr,R_b)
    #
    #
    #
    #                     all_bc_learn_time += t[0]
    #                     all_counter_example_time += t[1]
    #                     all_verify_time += t[2]
    #
    #                     if vis:
    #                         print(f"controller iteration: {i + 1}")
    #                         print(f"all iteration ddpg train time: {all_train_time}")
    #                         print(f"all iteration learn time: {all_bc_learn_time}")
    #                         print(f"all iteration counter examples generate time: {all_counter_example_time}")
    #                         print(f"all iteration sos verify time: {all_verify_time}")
    #                         break
    #
    #                 print(f"hidden_neurons:{b1_hidden_neurons}")
    #                 print(f"batch_size:{batch_size}")
    #                 print(f"lr:{lr}")
    #                 print(f"R_b:{R_b}")
    #                 print("-----------------------------------------")
    #
    #                 # 保存结果
    #                 results.append([i+1,iterbc,b1_hidden_neurons, batch_size, lr, R_b, all_train_time, all_bc_learn_time,
    #                                 all_counter_example_time, all_verify_time])
    #
    # # 转换为DataFrame
    # df = pd.DataFrame(results,
    #                   columns=['iter','iterbc','b1_hidden_neurons', 'batch_size', 'lr', 'R_b', 'all_train_time', 'all_bc_learn_time',
    #                            'all_counter_example_time', 'all_verify_time'])
    #
    # # 输出结果并保存为csv文件
    # print(df)
    # df.to_csv('hyperparameter_tuning_results.csv', index=False)
