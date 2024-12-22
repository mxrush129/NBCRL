# 1.Introduction

Safe controller synthesis is crucial for safety-critical
applications. This paper presents a novel reinforcement learning
approach to synthesize safe controllers for NN-controlled systems.
The core idea leverages an iterative scheme that combines safe
controller synthesis with neural barrier certificate (BC) synthe-
sis, ultimately producing a provably safe deep neural network
(DNN) controller with formal safety guarantees. The process
begins by pre-training a well-performing DNN controller as a
foundation via deep reinforcement learning (DRL). To formally
verify the safety properties of the closed-loop system under the
base controller, we propose a verification process that integrates
polynomial inclusion computations with neural BC synthesis. In
cases where the base controller is insufficient to yield a real BC,
the current spurious one is incorporated as an additional penalty
term to reshape the RL reward function, guiding the iterative
refinement for new controllers. We implement an automated tool,
NBCRL, and experimental results demonstrate the benefits of our
method in terms of efficiency and scalability even for a nonlinear
system with dimension up to 12.

# 2.Operating Guide

You can run this tool by following the steps below:

1.First of all, the python environment we use is 3.10.

2.You need to install some packages for using it.

```python
pip install -r requirements.txt
```

3.Our tool relies on the Gurobi solver, for which you need to obtain a license. 
You can find it at: https://www.gurobi.com/solutions/gurobi-optimizer/

4.When you have all the environment ready, you can run the tool.

Let's take C1 as an example to illustrate its use.

<1>You need to add the definition of the dynamical system in `./benchmarks/Examples.py`.

<2>You need to adjust the parameters and run `./benchmarks/C1.py` to start the iteration to get a barrier certificate and verify it.

You can find the following code in `./benchmarks/C1.py`:

```python
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
    b1_activations = ['MUL']
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
        # todo
        'max_iter': 5
    }
    Config = CegisConfig(**opts)
    cegis = Cegis(Config)
    vis, barrier, t,current_loss = cegis.solve()
    return vis, barrier, t


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    example_name = 'C1'
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

```