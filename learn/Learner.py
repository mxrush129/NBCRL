import torch
from utils.Config import CegisConfig
from learn.net_continuous import Net


class Learner:
    def __init__(self, config: CegisConfig):
        self.net = Net(config)
        self.config = config

    def learn_for_continuous(self, data, opt):
        learn_loops = self.config.learning_loops
        margin = self.config.margin
        slope = 1e-3
        relu6 = torch.nn.ReLU6()
        optimizer = opt

        data_tensor = data

        for epoch in range(learn_loops):
            optimizer.zero_grad()

            b1_y, bl_1, b1_grad, bm1_y, b2_y = self.net(data_tensor)

            b1_y, bl_1, bm1_y, b2_y = b1_y[:, 0], bl_1[:, 0], bm1_y[:, 0], b2_y[:, 0]

            weight = self.config.loss_weight_continuous

            accuracy = [0] * 3
            ###########
            # loss 1
            p = b1_y
            accuracy[0] = sum(p > margin / 2).item() * 100 / len(b1_y)

            loss_1 = weight[0] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 2
            p = b1_grad - bm1_y * bl_1
            accuracy[1] = sum(p > margin / 2).item() * 100 / len(bl_1)

            loss_2 = weight[1] * (torch.relu(-p + margin) - slope * relu6(p - margin)).mean()
            ###########
            # loss 8
            p = b2_y
            accuracy[2] = sum(p < -margin / 2).item() * 100 / len(b2_y)

            loss_8 = weight[2] * (torch.relu(p + margin) - slope * relu6(-p - margin)).mean()
            ###########
            loss = loss_1 + loss_8 + loss_2
            result = True

            for e in accuracy:
                result = result and (e == 100)

            if epoch % (learn_loops // 10) == 0 or result:
                print(f'{epoch}->', end=' ')
                for i in range(len(accuracy)):
                    print(f'accuracy{i + 1}:{accuracy[i]}', end=', ')
                print(f'loss:{loss}')

            loss.backward()
            optimizer.step()
            if result:
                break
        return loss