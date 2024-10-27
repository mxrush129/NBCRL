import sys, os
import random
import time
import timeit
from copy import deepcopy
import pickle
import torch
import numpy as np
import sympy as sp
from sympy.physics.vector.printing import params

from utils.Config import CegisConfig
from Examples import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis
from RL.train_controller import train_by_ddpg, ddpg_init
import nni

def save_object(obj, filename):
    if not os.path.exists(filename):
        # 创建该文件的父目录
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


def train_barrier(example_name, controller, params):
    b1_activations = ['SKIP']
    b1_hidden_neurons = [params["hidden_neuron_num"]] * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)

    start = timeit.default_timer()
    opts = {
        'b_act': b1_activations,
        'b_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': [],
        "batch_size": params['batch_size'],
        'lr': params['lr'],
        'loss_weight_continuous': (1, 1, 1),
        'R_b': params['R_b'],
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        # todo
        'max_iter': 5
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    vis, barrier, t, loss = cegis.solve()
    return vis, barrier, t, loss

# @func_set_timeout(60)
def train():
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    params = nni.get_next_parameter()
    print(f"Received parameters from NNI: {params}")

    params = {
        "lr": 0.4726888550563076,
        "C": "C9",
        "batch_size": 350,
        "R_b": 0.15668311838443136,
        "hidden_neuron_num": 5
    }
    example_name = params["C"]

    iter = 3

    B = None
    all_train_time = 0
    all_bc_learn_time = 0
    all_counter_example_time = 0
    all_verify_time = 0

    start_time = time.time()
    print("DDPG initializing...")
    agent, env, replay_buffer = ddpg_init(example_name)
    print("DDPG training...")

    saved_ddpg_data_path = r"E:\programing\workspace\TCAD2024\TCAD\benchmarks\data\ddpg_data" + "\\" + example_name

    for i in range(iter):
        data_i = load_object(saved_ddpg_data_path + f"\data{i}.pkl")
        if not data_i:
            tmp_controller, train_time = train_by_ddpg(agent, env, replay_buffer)
            save_object((tmp_controller, train_time), saved_ddpg_data_path + f"\data{i}.pkl")
        else:
            tmp_controller, train_time = data_i

        print(f"temp_controller: {tmp_controller}\n train_time: {train_time}\n")

        all_train_time += train_time
        controller = [tmp_controller]

        vis, barrier, t, loss = train_barrier(example_name, controller, params)

        all_bc_learn_time += t[0]
        all_counter_example_time += t[1]
        all_verify_time += t[2]

        nni.report_intermediate_result(loss.item())

        if vis:
            print(f"controller iteration: {i + 1}")
            print(f"all iteration ddpg train time: {all_train_time}")
            print(f"all iteration learn time: {all_bc_learn_time}")
            print(f"all iteration counter examples generate time: {all_counter_example_time}")
            print(f"all iteration sos verify time: {all_verify_time}")
            break

        B = barrier
        env.update_barrier(B)

        # if time.time() - start_time > 120:
        #     raise TimeoutError("Timeout")

    nni.report_final_result(loss.item())

    if not vis:
        assert False

if __name__ == '__main__':
    train()


