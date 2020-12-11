from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.networks import SequentialNetwork, ValueFunction
from RL_framework.buffer import ReplayMemory, ProcessMinibatch
import wandb

wandb.init(project='framework')

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# General details
# ~~~~~~~~~~~~~~~
wandb.config.algorithm = 'DQN'
num_episodes = 5

gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 32,
          'target_update': 1000
          }

wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'Qnet_layers': [nn.Linear(obs_size, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, n_actions)],
                  'dueling_nodes': [obs_size, 64, 128, n_actions]
                  }

learning_rates = {'Qnet_lr': 5e-4}
wandb.config.update(learning_rates)

epsilon = {'eps_start': 1,
           'eps_end': 0.1,
           'eps_decay': 250,
           }
wandb.config.update(epsilon)

# Initialisation
# ~~~~~~~~~~~~~~
net = SequentialNetwork(network_layers['Qnet_layers'])
wandb.config.netork = net
Qnet = ValueFunction(net, optim.Adam, learning_rates['Qnet_lr'], torch.nn.SmoothL1Loss(), epsilon, target_net=True)

buffer = ReplayMemory(params['buffer_size'])

# Get training
# ~~~~~~~~~~~~
episode_rewards = []
total_step = 0
for episode in tqdm(range(num_episodes)):

    episode_reward = 0
    step = 0

    state = env.reset()

    terminal = False
    while terminal is False:

        action = Qnet.epsilon_greedy_action(state, episode)

        next_state, reward, terminal, _ = env.step(action)
        step += 1
        total_step += 1

        buffer.add(state, action, reward, next_state, terminal, step, None)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and len(buffer) >= params['minibatch_size']:
            minibatch = buffer.random_sample(params['minibatch_size'])
            transitions = ProcessMinibatch(minibatch, gamma)

            target = transitions.qlearning_target(Qnet.target_net)
            current_v = transitions.current_value(Qnet.net)
            Qnet.optimise(target, current_v)

        if total_step % params['target_update'] == 0:
            Qnet.update_target()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
