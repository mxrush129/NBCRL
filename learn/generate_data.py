import numpy as np
import torch

from utils.Config import CegisConfig
from RL.Env import Zones, Example


class Data:
    def __init__(self, config: CegisConfig):
        self.config = config
        self.ex: Example = config.example
        self.n = self.ex.n_obs

    def get_data(self, zone: Zones, batch_size):
        global s
        if zone.shape == 'box':
            times = 1 / (1 - self.config.R_b)
            s = np.clip((np.random.rand(batch_size, self.n) - 0.5) * times, -0.5, 0.5)
            center = (zone.low + zone.up) / 2
            s = s * (zone.up - zone.low) + center

        elif zone.shape == 'ball':
            s = np.random.randn(batch_size, self.n)
            s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(zone.r) for e in s])
            s = np.array(
                [e * np.random.random() ** (1 / self.n) if np.random.random() > self.config.C_b else e for e in s])
            s = s + zone.center

        # from matplotlib import pyplot as plt
        # plt.plot(s[:, :1], s[:, -1], '.')
        # plt.gca().set_aspect(1)
        # plt.show()
        return torch.Tensor(s)

    def x2dotx(self, X, f):
        f_x = []
        for x in X:
            f_x.append([f[i](*x) for i in range(self.n)])
        return torch.Tensor(f_x)

    def generate_data_for_continuous(self):
        batch_size = self.config.batch_size
        l1 = self.get_data(self.ex.D_zones, batch_size)
        I = self.get_data(self.ex.I_zones, batch_size)
        U = self.get_data(self.ex.U_zones, batch_size)

        l1_dot = self.x2dotx(l1, self.ex.f)

        return l1, I, U, l1_dot

    # TODO: 需要基于微分方程的向量场完成取点
    # 起点都从init出发, 然后对模拟出来的点分类到init和domain里;unsafe仍然随机一下
    def generate_data_by_trajectory(self):
        batch_size = self.config.batch_size
        U = self.get_data(self.ex.U_zones, batch_size)
        I, l1 = self.stimulate(self.get_data(self.ex.I_zones, batch_size))
        l1_dot = self.x2dotx(l1, self.ex.f)
        # 看一下生成的数据量
        # print(f"l1: {len(l1)}, I: {len(I)}, U: {len(U)}")
        # 过滤一下点数量， 保证都是batch_size 个点
        if len(l1) > batch_size:
            l1 = l1[:batch_size]
            l1_dot = l1_dot[:batch_size]
        if len(I) > batch_size:
            I = I[:batch_size]
        if len(U) > batch_size:
            U = U[:batch_size]
        print(f"l1: {len(l1)}, I: {len(I)}, U: {len(U)}")
        return l1, I, U, l1_dot

    def stimulate(self, begin_points, dt=0.01, steps=40):
        '''从I区域中的初始点出发，基于微分方程模拟轨迹，并将轨迹点分类到init和domain区域'''
        f = self.ex.f
        n = self.n

        I_points = []
        domain_points = []
        
        current_points = begin_points.clone()
        
        for _ in range(steps + 1):  # 包含初始点
            # 区域分类
            in_I_mask = self.is_in_I(current_points)
            I_batch = current_points[in_I_mask]
            domain_batch = current_points[~in_I_mask]
            
            # 保存数据
            if len(I_batch) > 0:
                I_points.append(I_batch)
            if len(domain_batch) > 0:
                domain_points.append(domain_batch)
            
            # 数值积分 (欧拉法)
            f_x = self.x2dotx(current_points, f)
            current_points = current_points + dt * f_x  # 更新到下一个状态

        # 合并结果张量
        I_data = torch.cat(I_points, dim=0) if I_points else torch.zeros((0, n))
        domain_data = torch.cat(domain_points, dim=0) if domain_points else torch.zeros((0, n))
        
        return I_data, domain_data

    def is_in_I(self, points):
        """判断点是否位于初始区域I内"""
        zone = self.ex.I_zones
        device = points.device
        dtype = points.dtype
        
        if zone.shape == 'box':
            # 转换为与points同设备的张量
            low = torch.tensor(zone.low, device=device, dtype=dtype)
            up = torch.tensor(zone.up, device=device, dtype=dtype)
            # 逐维度检查范围
            lower_check = torch.all(points >= low, dim=1)
            upper_check = torch.all(points <= up, dim=1)
            return lower_check & upper_check
            
        elif zone.shape == 'ball':
            # 计算球心距离
            center = torch.tensor(zone.center, device=device, dtype=dtype)
            radius_sq = zone.r # r在这个proj里就是代表半径的平方
            diff = points - center
            dist_sq = torch.sum(diff**2, dim=1)
            return dist_sq <= radius_sq
            
        else:
            raise ValueError(f"Unsupported zone type: {zone.shape}")
        
