import gym
import math
import torch
import numpy as np


class GymEnv:
    def __init__(self, env):
        self.env_name = env
        self.env = gym.make(env)
        self.obs_size = self.env.observation_space.shape[0]
        self.obs_high = torch.Tensor(self.env.observation_space.high)
        if str(self.env.action_space)[:8] == 'Discrete':
            self.action_size = self.env.action_space.n
        elif str(self.env.action_space)[:3] == 'Box':
            self.action_size = self.env.action_space.shape[0]
            self.action_high = self.env.action_space.high
            self.action_low = self.env.action_space.low

    def reward_func(self, state, action=None):

        if self.env_name == 'CartPole-v0':
            theta_threshold_radians = 12 * 2 * math.pi / 360
            x_threshold = 2.4
            x, _, theta, _ = state
            done = bool(x < -x_threshold
                        or x > x_threshold
                        or theta < -theta_threshold_radians
                        or theta > theta_threshold_radians)
            return 1 if not done else 0

        elif self.env_name == 'Pendulum-v0':
            def angle_normalize(x):
                return ((x + np.pi) % (2 * np.pi)) - np.pi

            th = np.arccos(np.clip(state[0],-1,1))
            thdot = state[2]
            max_torque = 2.
            u = np.clip(action, -max_torque, max_torque)[0]
            costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
            return -costs
