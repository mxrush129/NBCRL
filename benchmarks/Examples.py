import numpy as np

from RL.Env import Zones, Example, Env

pi = np.pi


class F:
    def __init__(self):
        pass

    def f1(self, x):
        # 1/(1+sinx^2):
        return -9.49845082e-01 * x ** 2 + 9.19717026e-01 * x ** 4 - 4.06137871e-01 * x ** 6 + 0.99899106

    def f2(self, x):
        # sinx/(1+sinx^2):
        return 9.78842244e-01 * x - 8.87441593e-01 * x ** 3 + 4.35351792e-01 * x ** 5

    def f3(self, x):
        # sin(x) * cos(x) / (1 + sin(x) ** 2): \
        return 9.70088125e-01 * x - 1.27188818 * x ** 3 + 6.16181488e-01 * x ** 5

    def f4(self, x):
        # cosx / (1 + sinx ^ 2):
        return -1.42907660e+00 * x ** 2 + 1.29010139e+00 * x ** 4 - 5.75414531e-01 * x ** 6 + 0.99857329

    def f5(self, x):
        # sinx
        return 9.87855464e-01 * x - 1.55267355e-01 * x ** 3 + 5.64266597e-03 * x ** 5

    def f6(self, x):
        # cosx
        return -4.99998744e-01 * x ** 2 + 4.16558586e-02 * x ** 4 - 1.35953076e-03 * x ** 6 + 0.99999998

    def f7(self, x):
        return 0.03717 * x ** 4 - 0.2335 * x ** 3 + 0.05392 * x ** 2 + 0.983 * x + 0.001217  # [0,pi] sinx

    def f8(self, x):
        return -0.007567 * x ** 5 + 0.05943 * x ** 4 - 0.0209 * x ** 3 - 0.4881 * x ** 2 - 0.002745 * x + 1  # [0,pi] cosx


fun = F()

examples = {

    51: Example(  # ex22
        n_obs=2,
        u_dim=1,
        # D_zones=[[-4, 4]] * 2,
        # I_zones=[[-3, -1]] + [[-3, -1]],
        # U_zones=[[2, 4]] + [[1, 3]],
        D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
        I_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
        U_zones=Zones('box', low=[2, 1], up=[4, 3]),
        f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] + u[0],
           lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],
        u=10,
        dense=4,
        units=20,
        name='C1'),

    52: Example(  # useful ex23
        n_obs=2,
        u_dim=1,
        # D_zones=[[-3, 3]] * 2,
        # I_zones=[[-1, -0.9]] + [[1, 1.1]],
        # U_zones=[[-2.75, -2.25]] + [[-1.75, -1.25]],
        D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
        I_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
        U_zones=Zones('box', low=[-2.75, -1.75], up=[-2.25, -1.25]),
        f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u[0],
           lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7)],
        u=10,
        dense=4,
        units=20,
        name='C2'),
    #
    53: Example(
        n_obs=2,
        u_dim=1,
        # u_dim=1=[[-2, 2]] * 2,
        # I_zones=[[-0.1, 0]] * 2,
        # U_zones=[[1.2, 1.3]] + [[-0.1, 0.1]],
        D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
        I_zones=Zones('box', low=[-0.1, -0.1], up=[0, 0]),
        U_zones=Zones('box', low=[1.2, -0.1], up=[1.3, 0.1]),
        f=[
            lambda x, u: x[1],
            lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u[0],
        ],
        u=10,
        dense=4,
        units=20,
        name='C3'),
    #
    54: Example(  # useful ex24
        n_obs=2,
        u_dim=1,
        # D_zones=[[-3, 3]] * 2,
        # I_zones=[[0.5, 1.5]] * 2,
        # U_zones=[[-1.6, -0.4]] * 2,
        D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
        I_zones=Zones('box', low=[0.5, 0.5], up=[1.5, 1.5]),
        U_zones=Zones('box', low=[-1.6, -1.6], up=[-0.4, -0.4]),
        f=[lambda x, u: x[1],
           lambda x, u: -0.5 * x[0] ** 2 - x[1] + u[0],
           ],

        u=10,
        dense=4,
        units=20,
        name='C4'),

    55: Example(  # useful ex25
        n_obs=2,
        u_dim=1,
        # D_zones=[[-2, -2]] * 2,
        # I_zones=[[1, 2]] + [[-0.5, 0.5]],
        # U_zones=[[-1.4, -0.6]] + [[-1.4, -0.6]],
        D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
        I_zones=Zones('box', low=[1, -0.5], up=[2, 0.5]),
        U_zones=Zones('box', low=[-1.4, -1.4], up=[-0.6, -0.6]),
        f=[lambda x, u: x[1] + u[0],
           lambda x, u: -x[0] + (1 / 3) * x[0] ** 3 - x[1],
           ],

        u=10,
        dense=4,
        units=20,
        name='C5'),

    56: Example(  # useful ex26
        n_obs=3,
        u_dim=1,
        # D_zones=[[-0.2, 0.2]] * 3,
        # I_zones=[[-0.1, 0.1]] * 3,
        # U_zones=[[-0.18, -0.15]] * 3,
        D_zones=Zones('box', low=[-0.2, -0.2], up=[0.2, 0.2]),
        I_zones=Zones('box', low=[-0.1, -0.1], up=[0.1, 0.1]),
        U_zones=Zones('box', low=[-0.18, -0.18], up=[-0.15, -0.15]),
        f=[lambda x, u: -x[1] + u[0],
           lambda x, u: -x[2],
           lambda x, u: -x[0] - 2 * x[1] - x[2] + x[0] ** 3,  ##--+
           ],

        u=10,
        dense=4,
        units=20,
        name='C6'),

    57: Example(  # useful ex27
        n_obs=3,
        u_dim=1,
        # D_zones=[[-4, 4]] * 3,
        # I_zones=[[-1, 1]] * 3,
        # U_zones=[[2, 3]] * 3,
        D_zones=Zones('box', low=[-4] * 3, up=[4] * 3),
        I_zones=Zones('box', low=[-1] * 3, up=[1] * 3),
        U_zones=Zones('box', low=[2] * 3, up=[3] * 3),
        f=[lambda x, u: x[2] + 8 * x[1],
           lambda x, u: -x[1] + x[2],
           lambda x, u: -x[2] - x[0] ** 2 + u[0],  ##--+
           ],

        u=10,
        dense=4,
        units=20,
        name='C7'),

    58: Example(  # useful ex28
        n_obs=4,
        u_dim=1,
        # D_zones=[[-4, 4]] * 4,
        # I_zones=[[-0.2, 0.2]] * 4,
        # U_zones=[[-3, -1]] * 4,
        D_zones=Zones('box', low=[-4] * 4, up=[4] * 4),
        I_zones=Zones('box', low=[-0.2] * 4, up=[0.2] * 4),
        U_zones=Zones('box', low=[-3] * 4, up=[-1] * 4),
        f=[lambda x, u: -x[0] - x[3] + u[0],
           lambda x, u: x[0] - x[1] + x[0] ** 2 + u[0],
           lambda x, u: -x[2] + x[3] + x[1] ** 2,
           lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],

        u=10,
        dense=4,
        units=20,
        name='C8'),

    59: Example(  # useful ex29
        n_obs=5,
        u_dim=1,
        # D_zones=[[-3, 3]] * 5,
        # I_zones=[[0.5, 1.5]] * 5,
        # U_zones=[[-2.6, -1.4]] * 5,
        D_zones=Zones('box', low=[-3] * 5, up=[3] * 5),
        I_zones=Zones('box', low=[0.5] * 5, up=[1.5] * 5),
        U_zones=Zones('box', low=[-2.6] * 5, up=[-1.4] * 5),
        f=[
            lambda x, u: -0.1 * x[0] ** 2 - 0.4 * x[0] * x[3] - x[0] + x[1] + 3 * x[2] + 0.5 * x[3],
            lambda x, u: x[1] ** 2 - 0.5 * x[1] * x[4] + x[0] + x[2],
            lambda x, u: 0.5 * x[2] ** 2 + x[0] - x[1] + 2 * x[2] + 0.1 * x[3] - 0.5 * x[4],
            lambda x, u: x[1] + 2 * x[2] + 0.1 * x[3] - 0.2 * x[4],
            lambda x, u: x[2] - 0.1 * x[3] + u[0]
        ],

        u=10,
        dense=4,
        units=20,
        name='C9'),

    510: Example(  # useful ex19
        n_obs=6,
        u_dim=1,
        # D_zones=[[-2, 2]] * 6,
        # I_zones=[[1, 2]] * 6,
        # U_zones=[[-1, -0.5]] * 6,
        D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
        I_zones=Zones('box', low=[1] * 6, up=[2] * 6),
        U_zones=Zones('box', low=[-1] * 6, up=[-0.5] * 6),
        f=[
            lambda x, u: x[0] * x[2],
            lambda x, u: x[0] * x[4],
            lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
            lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
            lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
            lambda x, u: 2 * x[1] * x[4] + u[0]
        ],

        u=10,
        dense=4,
        units=20,
        name='C10'),

    511: Example(  # where
        n_obs=6,
        u_dim=1,
        # D_zones=[[0, 10]] * 6,
        # I_zones=[[3, 3.1]] * 6,
        # U_zones=[[4, 4.1]] + [[4.1, 4.2]] + [[4.2, 4.3]] + [[4.3, 4.4]] + [[4.4, 4.5]] + [[4.5, 4.6]],
        D_zones=Zones('box', low=[0] * 6, up=[10] * 6),
        I_zones=Zones('box', low=[3] * 6, up=[3.1] * 6),
        U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6]),
        f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u[0],
           lambda x, u: -x[0] - x[1] + x[4] ** 3,
           lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
           lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
           lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
           lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
           ],

        u=10,
        dense=4,
        units=20,
        name='C11'),

    512: Example(  # where

        n_obs=7,
        u_dim=1,
        # D_zones=[[-2, 2]] * 7,
        # I_zones=[[0.99, 1.01]] * 7,
        # U_zones=[[1.8, 2]] * 7,
        D_zones=Zones('box', low=[-2] * 7, up=[2] * 7),
        I_zones=Zones('box', low=[0.99] * 7, up=[1.01] * 7),
        U_zones=Zones('box', low=[1.8] * 7, up=[2] * 7),
        f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
           lambda x, u: 0.4 * x[0] - x[1],
           lambda x, u: x[1] - 5 * x[2] * x[3],
           lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
           lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
           lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
           lambda x, u: -0.5 * x[6] + u[0],
           ],

        u=10,
        dense=4,
        units=20,
        name='C12'),

    513: Example(  # useful ex17
        n_obs=9,
        u_dim=1,
        # D_zones=[[-2, 2]] * 9,
        # I_zones=[[0.99, 1.01]] * 9,
        # U_zones=[[1.8, 2]] * 9,
        D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
        I_zones=Zones('box', low=[0.99] * 9, up=[1.01] * 9),
        U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
        f=[
            lambda x, u: 3 * x[2] + u[0],
            lambda x, u: x[3] - x[1] * x[5],
            lambda x, u: x[0] * x[5] - 3 * x[2],
            lambda x, u: x[1] * x[5] - x[3],
            lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
            lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
            lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
            lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
            lambda x, u: 2 * x[5] * x[7] - x[8]
        ],

        u=10,
        dense=4,
        units=20,
        name='C13'),

    514: Example(  # useful ex18
        n_obs=12,
        u_dim=1,
        # D_zones=[[-2, 2]] * 12,
        # I_zones=[[-0.1, 0.1]] * 12,
        # U_zones=[[0, 0.5]] * 3 + [[0.5, 1.5]] * 4 + [[-1.5, -0.5]] + [[0.5, 1.5]] * 2 + [[-1.5, -0.5]] + [[0.5, 1.5]],
        D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
        I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
        U_zones=Zones('box', low=[0] * 3 + [0.5] * 4 + [-1.5] + [0.5] * 2 + [-1.5, 0.5],
                      up=[0.5] * 3 + [1.5] * 4 + [-0.5] + [1.5] * 2 + [-0.5, 1.5]),
        f=[
            lambda x, u: x[3],
            lambda x, u: x[4],
            lambda x, u: x[5],
            lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
            lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
            lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
            lambda x, u: x[9],
            lambda x, u: x[10],
            lambda x, u: x[11],
            lambda x, u: 9.81 * x[1],
            lambda x, u: -9.81 * x[0],
            lambda x, u: -16.3541 * x[11] + u[0]
        ],

        u=10,
        dense=4,
        units=20,
        name='C14'),
    515:Example(
            n_obs=3,
            u_dim=1,
            D_zones=Zones(shape='box', low=[-5] * 3, up=[5] * 3),
            I_zones=Zones(shape='ball', center=[-0.75, -1, -0.4], r=0.1 ** 2),
            G_zones=Zones(shape='ball', center=[0, 0, 0], r=0.1 ** 2),
            U_zones=Zones(shape='ball', center=[-0.3, -0.36, 0.2], r=0.1 ** 2, inner=True),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u[0],  ##--+
               ],
            u=3,
            dense=5,
            units=50,
            name='Academic 3D'
        ),  # Academic 3D

    ################################################################
    # 10: Example(
    #     n_obs=2,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
    #     I_zones=Zones('ball', center=[1, 1], r=1),
    #     G_zones=Zones('ball', center=[1, 1], r=0.25),
    #     U_zones=Zones('ball', center=[-1, -1], r=0.64),
    #     f=[lambda x, u: x[1],
    #        lambda x, u: -0.5 * x[0] ** 2 - x[1] + u[0],
    #        ],
    #     u=10,
    #     dense=4,
    #     name='c1',
    #     units=20,),
    # 7: Example(
    #     n_obs=2,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
    #     I_zones=Zones('box', low=[-0.1, -0.1], up=[0.6, 0]),
    #     G_zones=Zones('box', low=[-0.1, -0.1], up=[0, 0]),
    #     U_zones=Zones('box', low=[0.7, -0.1], up=[1.3, 0.1]),
    #     f=[lambda x, u: x[1],
    #        lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u[0],
    #        ],
    #     u=2,
    #     dense=5,
    #     name='c2',
    #     units=30,),
    #
    # 15: Example(
    #     n_obs=2,
    #     u_dim=1,
    #     D_zones=Zones(shape='box', low=[-6] * 2, up=[6] * 2),
    #     I_zones=Zones(shape='box', low=[-2.5] * 2, up=[-2] * 2),
    #     G_zones=Zones(shape='ball', center=[0, 0], r=0.1 ** 2),
    #     U_zones=Zones(shape='box', low=[4] * 2, up=[6] * 2),
    #     f=[lambda x, u: x[1],
    #        lambda x, u: -0.6 * x[1] - x[0] - x[0] ** 3 + u[0]
    #        ],
    #     u=2,
    #     dense=4,
    #     name='c3',
    #     units=20,),
    # 9: Example(
    #     n_obs=2,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
    #     I_zones=Zones('ball', center=[1, 0], r=0.09),
    #     G_zones=Zones('ball', center=[1, 0], r=0.25),
    #     U_zones=Zones('ball', center=[-1, 1], r=0.09),
    #     f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
    #        lambda x, u: x[1] * u[0],
    #        ],
    #     u=10,
    #     dense=4,
    #     name='c4',
    #     units=20,),
    #
    #
    # 6: Example(
    #     n_obs=2,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
    #     I_zones=Zones('box', low=[-3, -3], up=[1.1, 1]),
    #     G_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
    #     U_zones=Zones('ball', center=[3, 2], r=2 ** 2),
    #     f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] * u[0],
    #        lambda x, u: -2 * x[1] - x[0] ** 2 + u[0]],
    #     # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
    #     #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
    #     u=1,
    #     dense=4,
    #     name='c5',
    #     units=20,),
    # 1: Example(
    #     n_obs=2,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
    #     I_zones=Zones('box', low=[-0.51, 0.49], up=[-0.49, 0.51]),
    #     G_zones=Zones('box', low=[-0.05, -0.05], up=[0.05, 0.05]),
    #     U_zones=Zones('box', low=[-0.4, 0.2], up=[0.1, 0.35]),
    #     f=[lambda x, u: x[1],
    #        lambda x, u: (1 - x[0] ** 2) * x[1] - x[0] + u[0]
    #        ],  # 0.01177-3.01604*x1-19.59416*x2+2.96065*x1^2+27.86854*x1*x2+48.41103*x2^2
    #     u=3,
    #     dense=5,
    #     units=64,
    #     dt=0.005,
    #     max_episode=1500,
    #     name='c6'),
    #
    # 2: Example(
    #     n_obs=3,
    #     u_dim=1,
    #     D_zones=Zones(shape='box', low=[-5] * 3, up=[5] * 3),
    #     I_zones=Zones(shape='ball', center=[-0.75, -1, -0.4], r=0.35 ** 2),
    #     G_zones=Zones(shape='ball', center=[0, 0, 0], r=0.1 ** 2),
    #     U_zones=Zones(shape='ball', center=[-0.3, -0.36, 0.2], r=0.35 ** 2, inner=True),
    #     f=[lambda x, u: x[2] + 8 * x[1],
    #        lambda x, u: -x[1] + x[2],
    #        lambda x, u: -x[2] - x[0] ** 2 + u[0],  ##--+
    #        ],
    #     u=1,
    #     dense=5,
    #     name='c7',
    #     units=50,),
    # 3: Example(
    #     n_obs=4,
    #     u_dim=1,
    #     D_zones=Zones('ball', center=[0, 0, 0, 0], r=5),
    #     I_zones=Zones('ball', center=[0, 0, 0, 0], r=0.5),
    #     G_zones=Zones('ball', center=[0, 0, 0, 0], r=0.1),
    #     U_zones=Zones('ball', center=[1.5, 1.5, -1.5, -1.5], r=0.5),
    #     f=[lambda x, u: x[2],
    #        lambda x, u: x[3],
    #        lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u[0],
    #        lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
    #     # B=None,  ## 没有障碍函数写 None
    #     u=1,
    #     dense=5,
    #     name='c8',
    #     units=30,),
    # 8: Example(
    #     n_obs=4,
    #     u_dim=1,
    #     D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
    #     I_zones=Zones('box', low=[-0.8, -0.8, -0.8, -0.8], up=[0.2, 0.2, 0.2, 0.2]),
    #     G_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
    #     U_zones=Zones('ball', center=[-2, -2, -2, -2], r=1),
    #     f=[lambda x, u: -x[0] - x[3] + u[0],
    #        lambda x, u: x[0] - x[1] + x[0] ** 2 + u[0],
    #        lambda x, u: -x[2] + x[3] + x[1] ** 2,
    #        lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
    #     u=1,
    #     dense=5,
    #     name='c9',
    #     units=30,),
    # 13: Example(
    #     n_obs=5,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-3, -3, -3, -3, -3], up=[3, 3, 3, 3, 3]),
    #     I_zones=Zones('ball', center=[1, 1, 1, 1, 1], r=0.25),
    #     G_zones=Zones('ball', center=[1, 1, 1, 1, 1], r=0.25),
    #     U_zones=Zones('ball', center=[-2, -2, -2, -2, -2], r=0.36),
    #     f=[lambda x, u: -0.1 * x[0] ** 2 - 0.4 * x[0] * x[3] - x[0] + x[1] + 3 * x[2] + 0.5 * x[3],
    #        lambda x, u: x[1] ** 2 - 0.5 * x[1] * x[4] + x[0] + x[2],
    #        lambda x, u: 0.5 * x[2] ** 2 + x[0] - x[1] + 2 * x[2] + 0.1 * x[3] - 0.5 * x[4],
    #        lambda x, u: x[1] + 2 * x[2] + 0.1 * x[3] - 0.2 * x[4],
    #        lambda x, u: x[2] - 0.1 * x[3] + u[0]
    #        ],
    #     u=1,
    #     dense=5,
    #     name='c10',
    #     units=30,),
    #
    # 5: Example(
    #     n_obs=6,
    #     u_dim=6,
    #     D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
    #     I_zones=Zones('ball', center=[0] * 6, r=0.2 ** 2),
    #     U_zones=Zones('box', low=[-1] * 6, up=[-0.5] * 6),
    #     G_zones=Zones('ball', center=[0] * 6, r=0.1 ** 2),
    #     f=[lambda x, u: x[0] * x[2],
    #        lambda x, u: x[0] * x[4],
    #        lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
    #        lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
    #        lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
    #        lambda x, u: 2 * x[1] * x[4] + u[0]
    #        ],
    #     u=3,
    #     dense=5,
    #     name='c11',
    #     units=30,),
    #
    #
    #
    #
    # 11: Example(
    #     n_obs=6,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[0, 0, 0, 0, 0, 0], up=[10, 10, 10, 10, 10, 10]),
    #     I_zones=Zones('box', low=[3, 3, 3, 3, 3, 3], up=[3.1, 3.1, 3.1, 3.1, 3.1, 3.1]),
    #     G_zones=Zones('box', low=[3, 3, 3, 3, 3, 3], up=[3.1, 3.1, 3.1, 3.1, 3.1, 3.1]),
    #     U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6]),
    #     f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u[0],
    #        lambda x, u: -x[0] - x[1] + x[4] ** 3,
    #        lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
    #        lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
    #        lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
    #        lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
    #        ],
    #     u=1,
    #     dense=5,
    #     name='c12',
    #     units=30,),
    # 12: Example(
    #     n_obs=7,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-2, -2, -2, -2, -2, -2, -2], up=[2, 2, 2, 2, 2, 2, 2]),
    #     I_zones=Zones('box', low=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    #                   up=[1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]),
    #     G_zones=Zones('box', low=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    #                   up=[1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]),
    #     U_zones=Zones('box', low=[1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8], up=[2, 2, 2, 2, 2, 2, 2]),
    #     f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
    #        lambda x, u: 0.4 * x[0] - x[1],
    #        lambda x, u: x[1] - 5 * x[2] * x[3],
    #        lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
    #        lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
    #        lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
    #        lambda x, u: -0.5 * x[6] + u[0]
    #        ],
    #     u=3,
    #     dense=5,
    #     name='c13',
    #     units=30,),
    #
    # 14: Example(
    #     n_obs=9,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
    #     I_zones=Zones('box', low=[0.99] * 9, up=[1.01] * 9),
    #     G_zones=Zones('box', low=[0.99] * 9, up=[1.01] * 9),
    #     U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
    #     f=[lambda x, u: 3 * x[2] + u[0],
    #        lambda x, u: x[3] - x[1] * x[5],
    #        lambda x, u: x[0] * x[5] - 3 * x[2],
    #        lambda x, u: x[1] * x[5] - x[3],
    #        lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
    #        lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
    #        lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
    #        lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
    #        lambda x, u: 2 * x[5] * x[7] - x[8]
    #        ],
    #     u=3,
    #     dense=5,
    #     name='c14',
    #     units=30,),
    #
    # 4: Example(
    #     n_obs=9,
    #     u_dim=1,
    #     D_zones=Zones('box', low=[-5] * 9, up=[5] * 9),
    #     I_zones=Zones('box', low=[-0.1] * 9, up=[0.1] * 9),
    #     U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
    #     G_zones=Zones('ball', center=[0] * 9, r=0.1 ** 2),
    #     f=[lambda x, u: 3 * x[2] - x[0] * x[5],  # + u[0],
    #        lambda x, u: x[3] - x[1] * x[5],
    #        lambda x, u: x[0] * x[5] - 3 * x[2],
    #        lambda x, u: x[1] * x[5] - x[3],
    #        lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
    #        lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
    #        lambda x, u: 5 * x[3] + x[1] - 0.5 * x[6],
    #        lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
    #        lambda x, u: 2 * x[5] * x[7] - x[8]
    #        ],
    #     u=3,
    #     dense=5,
    #     name='c15',
    #     units=30,),

    # inner=False的样例
    #     16: Example(
    #         n_obs=2,
    #         u_dim=1,
    #         D_zones=Zones(shape='box', low=[-pi, -5], up=[pi, 5]),
    #         I_zones=Zones(shape='ball', center=[0, 0], r=2 ** 2),
    #         G_zones=Zones(shape='ball', center=[0, 0], r=0.1 ** 2),
    #         U_zones=Zones(shape='ball', center=[0, 0], r=2.5 ** 2, inner=False),
    #         f=[lambda x, u: x[1],
    #            lambda x, u: -10 * (0.005621 * x[0] ** 5 - 0.1551 * x[0] ** 3 + 0.9875 * x[0]) - 0.1 * x[1] + u[0]
    #            ],
    #         u=2,
    #         dense=4,
    #         name='test15',
    #         units=20,),
    #     17: Example(
    #         n_obs=7,
    #         u_dim=1,  ###mx: [0.86893033 0.36807829 0.55860075 2.75415022 0.22084824 0.08408990.27414744]
    #         D_zones=Zones('box', low=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) - 5,
    #                       up=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) + 5),
    #         I_zones=Zones('box', low=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) - 0.05,
    #                       up=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) + 0.05),
    #         G_zones=Zones('box', low=np.array([0.87, 0.37, 0.56, 2.75, 0.22, 0.08, 0.27]) - 0.1,
    #                       up=np.array([0.87, 0.37, 0.56, 2.75, 0.22, 0.08, 0.27]) + 0.1),
    #         U_zones=Zones('box', low=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) - 4.5,
    #                       up=np.array([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45]) + 4.5, inner=False),
    #         f=[lambda x, u: 1.4 * x[2] - 0.9 * x[0],
    #            lambda x, u: 2.5 * x[4] - 1.5 * x[1] + u[0],
    #            lambda x, u: 0.6 * x[6] - 0.8 * x[1] * x[2],
    #            lambda x, u: 2 - 1.3 * x[2] * x[3],
    #            lambda x, u: 0.7 * x[0] - x[3] * x[4],
    #            lambda x, u: 0.3 * x[0] - 3.1 * x[5],
    #            lambda x, u: 1.8 * x[5] - 1.5 * x[1] * x[6],
    #            ],
    #         u=0.3,
    #         dense=5,
    #         name='test16',
    #         units=50,),
    #     18: Example(
    #         n_obs=4,
    #         u_dim=1,
    #         D_zones=Zones('box', low=[-6] * 4, up=[6] * 4),
    #         I_zones=Zones('box', low=[1.25, 2.35, 1.25, 2.35], up=[1.55, 2.45, 1.55, 2.45]),
    #         G_zones=Zones('box', low=[1.25, 2.35, 1.25, 2.35], up=[1.55, 2.45, 1.55, 2.45]),
    #         U_zones=Zones('box', low=[-5] * 4, up=[5] * 4, inner=False),
    #         f=[lambda x, u: x[1] + u[0],
    #            lambda x, u: (1 - x[0] ** 2) * x[1] - 2 * x[0] + x[2],
    #            lambda x, u: x[3] + u[0],
    #            lambda x, u: (1 - x[2] ** 2) * x[3] - 2 * x[2] + x[0]
    #            ],
    #         u=1,
    #         dense=5,
    #         name='test17',
    #         units=30,),
    #     19: Example(
    #         n_obs=4,
    #         u_dim=2,
    #         D_zones=Zones('box', low=[-200, -200, -200, -200], up=[200, 200, 200, 200]),
    #         I_zones=Zones('box', low=[-50, -50, 0, 0], up=[-25, -25, 0, 0]),
    #         G_zones=Zones('box', low=[-10, -10, -10, -10], up=[10, 10, 10, 10]),
    #         U_zones=Zones('box', low=[-100, -100, -200, -200], up=[100, 100, 200, 200], inner=False),
    #         f=[lambda x, u: x[2],
    #            lambda x, u: x[3],
    #            lambda x, u: 0.004375 ** 2 * x[0] + 2 * 0.004375 * x[3] - x[0] * 1.9143193144076605e-05 + u[0] / 500,
    #            lambda x, u: 0.004375 ** 2 * x[1] - 2 * 0.004375 * x[2] - x[1] * 1.9143193144076605e-05 + u[1] / 500
    #            ],
    #         u=200,
    #         dense=4,
    #         name='test18',
    #         units=50,),
    #     20: Example(
    #         n_obs=12,
    #         u_dim=1,
    #         D_zones=Zones('box', low=[-200] * 12, up=[200] * 12),
    #         I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
    #         U_zones=Zones('ball', center=[0] * 12, r=200 ** 2, inner=False),
    #         G_zones=Zones('ball', center=[0] * 12, r=0.1 ** 2),
    #         f=[lambda x, u: x[3],
    #            lambda x, u: x[4],
    #            lambda x, u: x[5],
    #            lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
    #            lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
    #            lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
    #            lambda x, u: x[9],
    #            lambda x, u: x[10],
    #            lambda x, u: x[11],
    #            lambda x, u: 9.81 * x[1],
    #            lambda x, u: -9.81 * x[0],
    #            lambda x, u: -16.3541 * x[11] + u[0]
    #            ],
    #
    #         u=3,
    #         dense=5,
    #         name='test19',
    #         units=30,),

    # 随机系数
    #     21: Example(
    #         n_obs=3,
    #         u_dim=1,
    #         D_zones=Zones(shape='box', low=[-2] * 3, up=[2] * 3),
    #         I_zones=Zones(shape='box', low=[0.35, 0.45, 0.25], up=[0.45, 0.55, 0.35]),
    #         G_zones=Zones(shape='box', low=[-0.032] * 3, up=[0.032] * 3),
    #         U_zones=None,
    #         f=[lambda x, u: x[2] ** 3 - x[1] + np.random.random() * 0.02 - 0.01,
    #            # 删掉
    #            lambda x, u: x[2],
    #            lambda x, u: u[0]
    #            ],
    #         u=2,
    #         dense=3,
    #         units=30,
    #         name='test20'),

    # 多项式拟合三角函数
    #     22: Example(
    #         n_obs=4,
    #         u_dim=1,
    #         D_zones=Zones(shape='box', low=[-1.3] * 4, up=[1.3] * 4),
    #         I_zones=Zones(shape='ball', center=[0, 0, 0, 0], r=0.7 ** 2),
    #         G_zones=Zones(shape='ball', center=[0, 0, 0, 0], r=0.1 ** 2),
    #         U_zones=Zones(shape='ball', center=[0, 0, 0, 0], r=1 ** 2, inner=False),
    #         f=[lambda x, u: x[2],
    #            lambda x, u: x[3],
    #            lambda x, u: u[0] * fun.f1(x[1]) + x[3] ** 2 * fun.f2(x[1]) - fun.f3(x[1]),
    #            lambda x, u: u[0] * fun.f4(x[1]) + x[3] ** 2 * fun.f3(x[1]) - 2 * fun.f2(x[1])
    #            ],
    #         u=1,
    #         dense=1,
    #         name='test21',
    #         units=40,),
    #     23: Example(
    #         n_obs=2,
    #         u_dim=1,
    #         D_zones=Zones(shape='box', low=[-0.8] * 2, up=[0.8] * 2),
    #         I_zones=Zones(shape='ball', center=[0, 0], r=0.5 ** 2),
    #         G_zones=Zones(shape='ball', center=[-0.2, 0], r=0.2 ** 2),
    #         U_zones=Zones(shape='ball', center=[0, 0], r=0.6 ** 2, inner=False),
    #         f=[lambda x, u: 6 * fun.f5(x[1]),
    #            lambda x, u: 6 * u[0] - (fun.f6(x[1]) / (1 - x[0]))],
    #         u=2,
    #         dense=4,
    #         name='test22',
    #         units=30,),
    #     24: Example(
    #         n_obs=6,
    #         u_dim=2,
    #         D_zones=Zones(shape='box', low=[-1] * 6, up=[1] * 6),
    #         I_zones=Zones(shape='ball', center=[0, 0, 0, 0, 0, 0], r=0.3 ** 2),
    #         G_zones=Zones(shape='ball', center=[0, 0, 0, 0, 0, 0], r=0.1 ** 2),
    #         U_zones=Zones(shape='ball', center=[0, 0, 0, 0, 0, 0], r=1 ** 2, inner=False),
    #         f=[lambda x, u: x[3],
    #            lambda x, u: x[4],
    #            lambda x, u: x[5],
    #            lambda x, u: -(u[0] + u[1]) * fun.f5(x[2]) / 0.1,
    #            lambda x, u: ((u[0] + u[1]) * fun.f6(x[2]) - 0.01) / 0.1,
    #            lambda x, u: u[0] - u[1]
    #            ],
    #         u=1,
    #         dense=1,
    #         name='test23',
    #         units=50),
    #     25: Example(
    #         n_obs=5,
    #         u_dim=1,
    #         D_zones=Zones('box', low=[-1, -1, 0, 0.1, 0.1], up=[1, 1, pi, 0.1, 0.1]),
    #         I_zones=Zones('box', low=[1, 0.5, 0, 0.1, 0.1], up=[1, 1, pi, 0.1, 0.1]),
    #         G_zones=Zones('box', low=[-0.1, -0.1, 0, 0.1, 0.1], up=[0.1, 0.1, pi, 0.1, 0.1]),
    #         U_zones=Zones('box', low=[-1, -1, 0, 0.1, 0.1], up=[-0.5, -0.5, pi, 0.1, 0.1]),
    #         f=[lambda x, u: -x[3] * fun.f7(x[2]),
    #            lambda x, u: x[3] * fun.f8(x[2]) - x[4],
    #            lambda x, u: -u[0],
    #            lambda x, u: 0,
    #            lambda x, u: 0
    #            ],
    #         u=10,
    #         dense=5,
    #         name='test24',
    #         units=50,),
    #
    #     26: Example(
    #         n_obs=2,
    #         u_dim=1,
    #         D_zones=Zones('box', low=[-6, -7 * pi / 10], up=[6, 7 * pi / 10]),
    #         I_zones=Zones('box', low=[-1, -pi / 16], up=[1, pi / 16]),
    #         G_zones=Zones('ball', center=[0, 0], r=0.1 ** 2),
    #         U_zones=Zones('box', low=[2, pi / 8], up=[5, pi / 2]),
    #         f=[lambda x, u: 0.005621 * x[1] ** 5 - 0.1551 * x[1] ** 3 + 0.9875 * x[1],  # sin(x[1]),
    #            lambda x, u: -u[0]
    #            ],  # 0.01177-3.01604*x1-19.59416*x2+2.96065*x1^2+27.86854*x1*x2+48.41103*x2^2
    #         u=3,
    #         dense=5,
    #         name='test25',
    #         units=30,),

}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError('The example {} was not found.'.format(name))


if __name__ == '__main__':
    example = examples[1]
    env = Env(examples[1])
    env.reward_gaussian = False
    x, y, r = [], [], []
    s, info = env.reset(2024)
    print(s)
    x.append(s[0])
    y.append(s[1])
    done, truncated = False, False
    while not done and not truncated:
        action = np.array([1])
        observation, reward, terminated, truncated, info = env.step(action)
        x.append(observation[0])
        y.append(observation[1])
        r.append(reward)

    # from controller.Plot import plot
    #
    # plot(env, x, y)
    # print(sum(r))
