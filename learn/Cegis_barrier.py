import timeit
import os
import torch
import numpy as np
from sympy.physics.units import current

from utils.Config import CegisConfig
from learn.generate_data import Data
from learn.Learner import Learner
from verify.SosVerify import SOS
from verify.CounterExampleFind import CounterExampleFinder
from plot.plot import Draw


class Cegis:
    def __init__(self, config: CegisConfig):
        self.config = config
        self.max_cegis_iter = config.max_iter

    def solve(self):
        import mosek

        with mosek.Env() as env:
            with env.Task() as task:
                task.optimize()
                print('Mosek can be used normally.')

        # if self.config.example.continuous:
        data = Data(self.config).generate_data_for_continuous()
        # else:
        #     data = Data(self.config).generate_data()
        learner = Learner(self.config)
        optimizer = torch.optim.AdamW(learner.net.parameters(), lr=self.config.lr)

        counter = CounterExampleFinder(self.config)

        t_learn = 0
        t_cex = 0
        t_sos = 0
        vis_sos = False
        barrier, bc = None, None
        for i in range(self.max_cegis_iter):
            print(f'iter:{i + 1}')
            t1 = timeit.default_timer()
            # if self.config.example.continuous:
            current_loss = learner.learn_for_continuous(data, optimizer)
            # else:
            #     learner.learn(data, optimizer)
            t2 = timeit.default_timer()
            t_learn += t2 - t1

            barrier = learner.net.get_barriers()
            print(f'B(x) = {barrier[0]}')
            # for poly in barrier:
            #     print(poly)

            t3 = timeit.default_timer()
            sos = SOS(self.config, barrier)

            vis, state = sos.verify_continuous()

            if vis:
                print('Continuous system SOS verification passed!')
                print(f'B(x) = {barrier[0]}')
                print(f'Total number of iterations:{i + 1}')
                vis_sos = True
                t4 = timeit.default_timer()
                t_sos += t4 - t3
                break

            t4 = timeit.default_timer()
            t_sos += t4 - t3

            t5 = timeit.default_timer()
            res = counter.find_counterexample_for_continuous(state, barrier)
            bc = counter.get_expr_for_continuous(barrier)

            t6 = timeit.default_timer()
            t_cex += t6 - t5

            data = self.merge_data(data, res)

            # if self.config.example.continuous:
            #     draw = Draw(self.config.example, barrier[0])
            #     draw.draw_continuous()

        end = timeit.default_timer()
        if vis_sos:
            print('Total learning time:{}'.format(t_learn))
            print('Total counter-examples generating time:{}'.format(t_cex))
            print('Total sos verifying time:{}'.format(t_sos))
            self.save_model(learner.net, barrier)
            # if self.config.example.n_obs == 2:
            #     draw = Draw(self.config.example, barrier[0])
            #     draw.draw_continuous()
            #     draw.draw_3d_continuous()
            return True, bc, (t_learn, t_cex, t_sos), current_loss
        else:
            print('Failed')
            return False, bc, (t_learn, t_cex, t_sos), current_loss

    def merge_data(self, data, add_res):
        res = []
        ans = 0
        for x, y in zip(data, add_res):
            if len(y) > 0:
                ans = ans + len(y)
                y = torch.Tensor(np.array(y))
                res.append(torch.cat([x, y], dim=0).detach())
            else:
                res.append(x)
        print(f'Add {ans} counterexamples!')
        return tuple(res)

    def save_model(self, net, barrier):
        if not os.path.exists('../model'):
            os.mkdir('../model')
        torch.save(net.state_dict(), f'../model/{self.config.example.name}.pt')

        if not os.path.exists('../result'):
            os.mkdir('../result')

        with open(f'../result/{self.config.example.name}.txt', 'w') as f:
            f.write(f'B={barrier[0]}')
