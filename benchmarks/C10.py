import random
import timeit
from copy import deepcopy

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


def train_barrier(example_name, controller):
    b1_activations = ['SKIP']
    b1_hidden_neurons = [15] * len(b1_activations)

    example = deepcopy(get_example_by_name(example_name))
    example.f = update_f(example, controller)

    start = timeit.default_timer()
    opts = {
        'b_act': b1_activations,
        'b_hidden': b1_hidden_neurons,
        "example": example,
        'bm1_act': [],
        'bm1_hidden': [],
        "batch_size": 1500,
        'lr': 0.1,
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
    # print(cegis.solve())
    vis, barrier, t,loss = cegis.solve()
    return vis, barrier, t


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    

    b1_hidden_neurons_values = [5, 10, 15, 20, 50]
    batch_sizes = [500, 1000, 1500, 2000, 5000]
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    R_b_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []
    
    example_name = 'C10'
    
    iter = 3
    B = None
    all_train_time = 0
    all_bc_learn_time = 0
    all_counter_example_time = 0
    all_verify_time = 0

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
                        x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6')
                        controller = [(0.79197406661727) + (0.7524151033938165) * x1 - (1.6437397668244673) * x2 - (
                            0.6623515384444854) * x3 + (0.16969669798446635) * x4 - (0.002028035869638553) * x5 + (
                                          0.828796129580916) * x6 - (0.45091121582196747) * x1 ** 2 + (
                                          0.5458811914838341) * x1 * x2 + (0.1904703306476659) * x1 * x3 - (
                                          0.3253739143355701) * x1 * x4 + (0.1266441582232395) * x1 * x5 - (
                                          0.38323522194374815) * x1 * x6 - (0.8739983067545324) * x2 ** 2 - (
                                          0.197305927389085) * x2 * x3 + (0.7624630146376018) * x2 * x4 + (
                                          1.7789361518742839) * x2 * x5 + (0.7977319579603613) * x2 * x6 + (
                                          0.023561444373482163) * x3 ** 2 + (0.12090350200825938) * x3 * x4 + (
                                          0.02747067197214026) * x3 * x5 + (0.5885095594464359) * x3 * x6 + (
                                          0.10255274138847068) * x4 ** 2 - (0.5953075404988596) * x4 * x5 - (
                                          0.6003665372501428) * x4 * x6 - (0.15909262528028434) * x5 ** 2 - (
                                          1.072334941141975) * x5 * x6 - (0.23972601050684064) * x6 ** 2 - (
                                          0.0347213497014785) * x1 ** 3 + (0.003516201256483) * x1 ** 2 * x2 + (
                                          0.03250715127821646) * x1 ** 2 * x3 + (0.05007596888757762) * x1 ** 2 * x4 - (
                                          0.007235140783247322) * x1 ** 2 * x5 + (0.2953971226499105) * x1 ** 2 * x6 - (
                                          0.03851834818245897) * x1 * x2 ** 2 + (0.05390267755811967) * x1 * x2 * x3 - (
                                          0.13955835730159807) * x1 * x2 * x4 + (
                                          0.0007386004534625235) * x1 * x2 * x5 - (
                                          0.18530551700149206) * x1 * x2 * x6 - (
                                          0.004160142700518676) * x1 * x3 ** 2 - (
                                          0.011519638757403583) * x1 * x3 * x4 - (
                                          0.06771521672639468) * x1 * x3 * x5 - (0.14458860081883196) * x1 * x3 * x6 - (
                                          0.026095357936607666) * x1 * x4 ** 2 + (
                                          0.08507675981740367) * x1 * x4 * x5 + (0.18977763684189833) * x1 * x4 * x6 - (
                                          0.03805071017235773) * x1 * x5 ** 2 - (0.05903121370258191) * x1 * x5 * x6 - (
                                          0.07224719025811749) * x1 * x6 ** 2 + (0.06742956396691341) * x2 ** 3 + (
                                          0.17180978943223707) * x2 ** 2 * x3 + (0.07851102237344043) * x2 ** 2 * x4 + (
                                          0.035642430035781436) * x2 ** 2 * x5 + (
                                          0.12369208301649542) * x2 ** 2 * x6 - (
                                          0.003502248184862944) * x2 * x3 ** 2 + (
                                          0.008993662582307604) * x2 * x3 * x4 - (0.16264808326794) * x2 * x3 * x5 - (
                                          0.10796883817736236) * x2 * x3 * x6 - (0.11259482361431639) * x2 * x4 ** 2 - (
                                          0.1151558010343848) * x2 * x4 * x5 - (0.14110317850713106) * x2 * x4 * x6 - (
                                          0.19788421194997602) * x2 * x5 ** 2 - (0.5139859433045619) * x2 * x5 * x6 + (
                                          0.10010878448114491) * x2 * x6 ** 2 - (0.0071507696103521035) * x3 ** 3 + (
                                          0.05701923277130705) * x3 ** 2 * x4 - (
                                          0.019740715926270386) * x3 ** 2 * x5 - (
                                          0.021639781463672292) * x3 ** 2 * x6 - (0.0969743546427334) * x3 * x4 ** 2 + (
                                          0.12878539376187714) * x3 * x4 * x5 - (
                                          0.055406566071126595) * x3 * x4 * x6 - (
                                          0.06290593139336453) * x3 * x5 ** 2 + (0.24845725316733264) * x3 * x5 * x6 - (
                                          0.1360040523570416) * x3 * x6 ** 2 + (0.09239398412372424) * x4 ** 3 - (
                                          0.07530583448561656) * x4 ** 2 * x5 - (
                                          0.013869554392409944) * x4 ** 2 * x6 + (
                                          0.30351272780300564) * x4 * x5 ** 2 - (0.13240510244453574) * x4 * x5 * x6 + (
                                          0.2591348869951643) * x4 * x6 ** 2 - (0.1536949496334729) * x5 ** 3 + (
                                          0.5542623487375353) * x5 ** 2 * x6 + (0.03327911814693471) * x5 * x6 ** 2 - (
                                          0.009218046567513627) * x6 ** 3]
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



                    results.append([i+1,b1_hidden_neurons, batch_size, lr, R_b, all_train_time, all_bc_learn_time,
                                    all_counter_example_time, all_verify_time])


    df = pd.DataFrame(results,
                      columns=['iter','b1_hidden_neurons', 'batch_size', 'lr', 'R_b', 'all_train_time', 'all_bc_learn_time',
                               'all_counter_example_time', 'all_verify_time'])


    print(df)
    df.to_csv('hyperparameter_tuning_results.csv', index=False)

