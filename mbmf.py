from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.common.networks import SequentialNetwork
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.utils import *
import wandb
import math
import numpy as np
import matplotlib.pyplot as plt


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

def reward_func(state):
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    x, _, theta, _ = state
    done = bool(x < -x_threshold
                or x > x_threshold
                or theta < -theta_threshold_radians
                or theta > theta_threshold_radians)
    return 1 if not done else 0

# General details
# ~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'MBMF'
num_episodes = 100

gamma = 0.99
params = {'sample_collection': 10,
          'buffer_size': 2000,
          'minibatch_size': 256,
          'random_buffer_size': 2000,
          'training_epoch': 50,
          'control_horizon': 20,
          'K_actions_sample': 50,
          'dataset_ratio': 0.9
}
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'model_layers': [nn.Linear(obs_size + 1, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, obs_size)]
                  }
learning_rates = dict(model_lr=1E-3)
loss_fnc = torch.nn.MSELoss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

def K_rollouts(state, K, horizon):
    samples = np.random.randint(n_actions, size=(K, horizon))
    rewards = np.zeros(K)
    for i, sample in enumerate(samples):
        current_state = torch.Tensor(state)
        reward = 0
        for i, action in enumerate(sample):
            state_action = torch.cat((current_state, torch.Tensor([action])))
            current_state += model(state_action)
            reward += gamma**i * reward_func(current_state)
        rewards[i] = reward
    best_K = np.argmax(rewards)
    return samples[best_K, 0]

# Initialisation
# ~~~~~~~~~~~~~~
model = SequentialNetwork(network_layers['model_layers'])
opt = optim.Adam(model.parameters(), lr=learning_rates['model_lr'])

dataset_random = ReplayMemory(params['random_buffer_size'])
dataset_rl = ReplayMemory(params['buffer_size'])

# Gather random data and train dynamics models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while len(dataset_random) < params['random_buffer_size']:
    state = env.reset() + torch.normal(0, 0.001, size=(obs_size,)).numpy()
    terminal = False
    while terminal is False:
        action = np.random.randint(n_actions)
        next_state, reward, terminal, _ = env.step(action)
        dataset_random.add(state, action, reward, next_state, terminal, None, None)
        state = next_state

losses = []
for i in range(params['training_epoch']):
    minibatch = dataset_random.random_sample(params['minibatch_size'])
    t = ProcessMinibatch(minibatch)
    t.standardise()
    target = t.next_states - t.states + torch.normal(0, 0.001, size=t.states.shape)
    state_actions = torch.cat((t.states, t.actions), dim=1)
    current = model(state_actions + torch.normal(0, 0.001, size=state_actions.shape))
    loss = loss_fnc(target, current)
    losses.append(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
plt.plot(list(range(params['training_epoch'])), losses)
plt.show()

# Model based controller loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
episode_rewards = []
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    step = 0
    state = env.reset()
    terminal = False
    while terminal is False:
        action = K_rollouts(state, params['K_actions_sample'], params['control_horizon'])
        next_state, reward, terminal, _ = env.step(action)
        step += 1
        dataset_rl.add(state, action, reward, next_state, terminal, step, None)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and\
                len(dataset_rl) >= params['minibatch_size']:

            minibatch_random = dataset_random.random_sample(round(params['minibatch_size'] * (1-params['dataset_ratio'])))
            minibatch_rl = dataset_rl.random_sample(round(params['minibatch_size'] * params['dataset_ratio']))
            t_random = ProcessMinibatch(minibatch_random)
            t_rl = ProcessMinibatch(minibatch_rl)
            t_random.standardise()
            t_rl.standardise()
            target = torch.cat((t_random.next_states, t_rl.next_states)) - torch.cat((t_random.states, t_rl.states))
            state_actions = torch.cat((torch.cat((t_random.states, t_rl.states)),
                                       torch.cat((t_random.actions, t_rl.actions))), dim=1)
            current = model(state_actions)
            loss = loss_fnc(target, current)
            wandb.log({"model_loss": loss}, commit=False)
            opt.zero_grad()
            loss.backward()
            opt.step()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
