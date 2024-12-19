import numpy as np
import sympy as sp


class Zones:  
    def __init__(self, shape: str, center=None, r=None, low=None, up=None, inner=True):
        self.shape = shape
        self.inner = inner
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2 
            self.r = sum(((self.up - self.low) / 2) ** 2)  
        else:
            raise NotImplementedError


class Example:
    def __init__(self, n_obs, u_dim, D_zones, I_zones, f, u, dense, units, name, dt=0.001, G_zones=None,
                 U_zones=None, goal='avoid', max_episode=1000):
        self.n_obs = n_obs  
        self.u_dim = u_dim  
        self.D_zones = D_zones 
        self.I_zones = I_zones 
        self.G_zones = G_zones 
        self.U_zones = U_zones 
        self.f = f  
        self.u = u  
        self.dense = dense  
        self.units = units  
        # self.activation = activation  
        # self.k = k  
        self.name = name  
        self.dt = dt  
        self.goal = goal  # 'avoid','reach','reach-avoid'
        self.max_episode = max_episode


class Env:
    def __init__(self, example: Example):
        self.n_obs = example.n_obs
        self.u_dim = example.u_dim
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.G_zones = example.G_zones
        self.U_zones = example.U_zones
        self.f = example.f
        # self.path = example.path
        self.u = example.u

        self.dense = example.dense  
        self.units = example.units  
        # self.activation = example.activation  
        self.name = example.name
        self.dt = example.dt
        self.goal = example.goal
        self.s = None
        self.beyond_domain = True
        self.episode = 0
        self.max_episode = example.max_episode
        self.reward_gaussian = True
        self.barrier = None

    def update_barrier(self, barrier):
        self.barrier = barrier

    def reset(self):
        if self.I_zones.shape == 'ball':
            state = np.random.randn(self.n_obs)
            state = state / np.sqrt(sum(state ** 2)) * self.I_zones.r * np.random.random() ** (1 / self.n_obs)
            state = state + self.I_zones.center
        else:
            state = np.random.rand(self.n_obs) - 0.5
            state = state * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        self.s = state
        self.episode = 0
        return state, 'Reset successfully!'

    def step(self, action):
        self.episode += 1
        ds = np.array([F(self.s, action) for F in self.f])
        self.s = self.s + ds * self.dt
        done = self.check_done()
        truncated = self.check_truncated()
        reward = self.get_reward()

        return self.s, reward, done, truncated, {'episode': self.episode}

    def get_reward(self):
        if self.goal == 'avoid':
            reward = self.avoid_reward()
        elif self.goal == 'reach':
            reward = self.reach_reward()
        else:
            reward = self.avoid_reward() + self.reach_reward()
        return reward

    def avoid_reward(self):
        if self.reward_gaussian:
            if self.U_zones.shape == 'box':
                reward_avoid = -np.exp(
                    -sum((self.s - self.U_zones.center) ** 2 / ((self.U_zones.up - self.U_zones.low) / 2) ** 2))
            else:
                reward_avoid = -np.exp(-sum((self.s - self.U_zones.center) ** 2 / self.U_zones.r ** 2))
        else:
            reward_avoid = np.sqrt(sum((self.s - self.U_zones.center) ** 2)) - self.U_zones.r

        if not self.U_zones.inner:
            reward_avoid = -reward_avoid
        reward_avoid -= self.get_feedback_reward()
        return reward_avoid

    def get_feedback_reward(self):
        if self.barrier is None:
            return 0
        else:
            reward = 0
            b, db = self.barrier
            x = sp.symbols([f'x{i + 1}' for i in range(self.n_obs)])
            f_b, f_db = sp.lambdify(x, b), sp.lambdify(x, db)
            if self.check_zone(self.I_zones) and f_b(*self.s) < 0:
                reward += abs(f_b(*self.s))
            if self.check_zone(self.U_zones) and f_b(*self.s) < 0:
                reward += abs(f_db(*self.s))
            if self.check_zone(self.D_zones) and f_db(*self.s) < 0:
                reward += abs(f_db(*self.s))
            # print(f'reward:{reward}')
            return reward

    def check_zone(self, zone: Zones):
        if zone.shape == 'box':
            vis = sum([zone.low[i] <= self.s[i] <= zone.up[i] for i in range(self.n_obs)]) != self.n_obs
        else:
            vis = sum((self.s - zone.center) ** 2) >= zone.r ** 2

        return vis ^ zone.inner

    def reach_reward(self):
        if self.reward_gaussian:
            if self.G_zones.shape == 'box':
                reward_reach = np.exp(
                    -sum((self.s - self.G_zones.center) ** 2 / ((self.G_zones.up - self.G_zones.low) / 2) ** 2))
            else:
                reward_reach = np.exp(-sum((self.s - self.G_zones.center) ** 2 / self.G_zones.r ** 2))
        else:
            reward_reach = self.G_zones.r - np.sqrt(sum((self.s - self.G_zones.center) ** 2))

        if not self.G_zones.inner:
            reward_reach = -reward_reach
        return reward_reach

    def check_done(self):
        done = False
        if self.U_zones.shape == 'box':
            vis = sum([self.U_zones.low[i] <= self.s[i] <= self.U_zones.up[i] for i in range(self.n_obs)]) != self.n_obs
        else:
            vis = sum((self.s - self.U_zones.center) ** 2) >= self.U_zones.r ** 2

        if (vis ^ self.U_zones.inner) and self.goal != 'reach':
            
            done = True

        
        if self.D_zones.shape == 'box':
            vis = sum([self.D_zones.low[i] <= self.s[i] <= self.D_zones.up[i] for i in range(self.n_obs)]) == self.n_obs
        else:
            vis = sum((self.s - self.D_zones.center) ** 2) <= self.D_zones.r ** 2

        if vis ^ self.D_zones.inner:
            
            if self.beyond_domain:
                done = True
            else:
                if self.D_zones.shape == 'box':
                    self.s = np.array(
                        [min(self.D_zones.up[i], max(self.D_zones.low[i], self.s[i])) for i in range(self.n_obs)])
                else:
                    ratio = self.D_zones.r / np.sqrt((self.s - self.D_zones.center) ** 2)
                    self.s = (self.s - self.D_zones.center) * ratio + self.D_zones.center
        return done

    def check_truncated(self):
        return self.episode >= self.max_episode
