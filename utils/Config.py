import torch


class CegisConfig:
    b_act = ['SKIP']
    b_hidden = [10]
    example = None
    bm_hidden = []
    bm_act = []
    batch_size = 500
    lr = 0.1
    loss_weight_continuous = (1, 1, 1)
    R_b = 0.5
    margin = 0.5
    DEG_continuous = [2] * 4
    split = False
    learning_loops = 100
    OPT = torch.optim.AdamW
    max_iter = 100
    bm = None
    counterexample_nums = 100
    lie_counterexample = 0
    counterexamples_ellipsoid = False
    eps = 0.05
    C_b = 0.2

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
