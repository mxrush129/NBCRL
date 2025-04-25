import sys, os
from pathlib import Path
proj_path = str(Path(__file__).resolve().parents[1])
sys.path.append(proj_path)
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
    b1_hidden_neurons = [5] * len(b1_activations)

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
        'lr': 0.2,
        'loss_weight_continuous': (1, 1, 1),
        'R_b': 0.6,
        'margin': 2,
        "DEG_continuous": [2, 2, 1, 2],
        "learning_loops": 100,
        # todo
        'max_iter': 20
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    
    vis, barrier, t,loss,_ = cegis.solve()
    return vis, barrier, t


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    example_name = 'C8'
    
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
        
        x1, x2, x3 ,x4= sp.symbols('x1 x2 x3 x4')
        controller=[0.9002150492055848-0.00539574752046955*x1-0.002528158271563848*x2+0.005438143596106565*x3-0.03914655258969325*x4-0.030822612244918574*x1**2+0.03171499190282263*x1*x2+0.08844077620797547*x1*x3-0.02360484198789681*x1*x4-0.02038397743994904*x2**2-0.03216759427153882*x2*x3+0.008267802649426706*x2*x4-0.03621692547016895*x3**2+0.10590532301940392*x3*x4-0.026794146353019105*x4**2-0.012791429437680083*x1**3-0.031690755113896636*x1**2*x2-0.03088264949629078*x1**2*x3+0.053268023781768224*x1**2*x4+0.04225539439063492*x1*x2**2-0.29730785342187943*x1*x2*x3+0.22336189623783448*x1*x2*x4+0.25646739518893086*x1*x3**2-0.4436574584123578*x1*x3*x4+0.19812247078899764*x1*x4**2+0.008354876869606398*x2**3+0.04123716107547222*x2**2*x3-0.026326827213556177*x2**2*x4+0.08166418476787823*x2*x3**2-0.27170841291057474*x2*x3*x4+0.16791418484202622*x2*x4**2-0.11615169322076341*x3**3+0.4120094399324352*x3**2*x4-0.4544483686248528*x3*x4**2+0.15697211945364373*x4**3]
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

