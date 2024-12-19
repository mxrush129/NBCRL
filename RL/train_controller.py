import re
import time
import timeit

import sympy as sp
import matplotlib.pyplot as plt
from RL.DDPG import DDPG
from RL.Env import Env
from benchmarks.Examples import get_example_by_name, Example
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
# from controller.Plot import plot
from RL.plot import plot
from RL.share import *
from collections import deque


def fit(env, agent):
    x, y = [], []
    # todo
    N = 10
    for i in range(N):
        state, info = env.reset()
        tot = 0
        mxlen=50
        dq=deque(maxlen=mxlen)
        while True:
            tot += 1
            if tot >= 2000:
                print('The {}th trajectory'.format(i))
                break
            dq.append(sum(env.s**2))

            if len(dq)==mxlen and np.var(dq)<1e-10:
                print('The {}th trajectory'.format(i))
                break
            action = agent.take_action(state)
            x.append(state)
            y.append(action)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            if done:
                print('The {}th unsafe trajectory'.format(i))
                break
    print(f"sample num：{len(x)}")
    P = PolynomialFeatures(2, include_bias=False)
    X = P.fit_transform(x)
    model = Ridge(alpha=0.00, fit_intercept=False)
    
    model.fit(X, y)


    s = ''
    for k, v in zip(P.get_feature_names_out(), model.coef_[0]):
        k = re.sub(r' ', r'*', k)
        k = k.replace('^', '**')
        if v < 0:
            s += f'- {-v} * {k} '
        else:
            s += f'+ {v} * {k} '

    x = sp.symbols(['x0', 'x1'])
    x_ = sp.symbols(['x1', 'x2'])
    temp = sp.sympify(s[1:])
    u = sp.lambdify(x, temp)(*x_)
    print(f'controller:{u}')
    return u


def ddpg_init(example_name):
    example = get_example_by_name(example_name)
    actor_lr = 3e-4
    critic_lr = 3e-3
    # todo
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    env = Env(example)
    env.reward_gaussian = False
    env.beyond_domain = False
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.n_obs
    action_dim = env.u_dim
    action_bound = env.u  # largest value of action
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return agent, env, replay_buffer


def train_by_ddpg(agent, env, replay_buffer):
    num_episodes = 20
    minimal_size = 1000
    batch_size = 64

    return_list = []
    start_time = timeit.default_timer()
    for i_episode in range(num_episodes):
        episode_return = 0
        state_list = []
        state, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            state_list.append(state)
            action = agent.take_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                   'dones': b_d}
                agent.update(transition_dict)
        return_list.append(episode_return)

        print(f'episode:{i_episode + 1},reward:{episode_return},step:{len(state_list)}')
        if i_episode % 10 == 0:
            state_list = np.array(state_list)
            x = state_list[:, :1]
            y = state_list[:, 1:2]
            plot(env, x, y)
    end_time = timeit.default_timer()
    controller = fit(env, agent)
    return controller, end_time - start_time


if __name__ == '__main__':
    example = get_example_by_name('Academic')

    agent, env, replay_buffer = ddpg_init('Academic')
    return_list = train_by_ddpg(agent,env,replay_buffer)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG')
