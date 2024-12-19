import copy
import json
import random
import timeit
from copy import deepcopy

import torch
import numpy as np
import sympy as sp
from utils.Config import CegisConfig
from benchmarks.Examples import get_example_by_name, get_example_by_id
from learn.Learner import Learner
from learn.Cegis_barrier import Cegis
from RL.train_controller import train_by_ddpg, ddpg_init


def update_f(example, u):
    x = sp.symbols([f'x{i + 1}' for i in range(example.n_obs)])
    f = [ff(x, u) for ff in example.f]
    return [sp.lambdify(x, ff) for ff in f]


def train_barrier(opts, example_name, controller):
    b1_activations = ['SKIP']
    b1_hidden_neurons = [10] * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)

    opts["b_act"] = b1_activations
    opts["b_hidden"] = b1_hidden_neurons
    opts["example"] = example

    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    vis, barrier, t = cegis.solve()
    return vis, barrier, t

import ray
ray.init(num_cpus=2)

@ray.remote
def solve(example_name, opts, iter=25):
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
        if i == 1:
            controller = [1]
        vis, barrier, t = train_barrier(opts, example_name, controller)
        all_bc_learn_time += t[0]
        all_counter_example_time += t[1]
        all_verify_time += t[2]

        if vis:
            from loguru import logger
            logger.remove()
            logger.add(example_name + "_{time}.log")
            logger.info(f"controller iteration: {i + 1}")
            logger.info(f"all iteration ddpg train time: {all_train_time}")
            logger.info(f"all iteration learn time: {all_bc_learn_time}")
            logger.info(f"all iteration counter examples generate time: {all_counter_example_time}")
            logger.info(f"all iteration sos verify time: {all_verify_time}")

            logger.info(f"batch_size: {opts['batch_size']}")
            logger.info(f"lr: {opts['lr']}")
            logger.info(f"max_iter: {opts['max_iter']}")

            break
        B = barrier
        env.update_barrier(B)


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    # todo
    example_name = 'C4'
    opts = {
        'bm1_act': [],
        "batch_size": 500,
        'lr': 0.2,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.8,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        # todo
        'max_iter': 5
    }

    configs = []

    for _lr in np.arange(0.05, 0.55, 0.05):
        _opts = copy.deepcopy(opts)
        _opts["lr"] = _lr
        configs.append(_opts)


    tasks = [solve.remote(example_name, _opts, 25) for _opts in configs]
    results = ray.get(tasks)
