import gym
import math
import torch

class gym_env:
    def __init__(self, env):
        self.env_name = env
        self.env = gym.make(env)
        self.obs_size = self.env.observation_space.shape[0]
        self.obs_max = torch.Tensor(self.env.observation_space.high)
        if str(self.env.action_space)[:8] == 'Discrete':
            self.action_size = self.env.action_space.n
        elif str(self.env.action_space)[:3] == 'Box':
            self.action_size = self.env.action_space.shape[0]

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
