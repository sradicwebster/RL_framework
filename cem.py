import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from RL_framework.common.networks import SequentialNetwork
from RL_framework.common.buffer import ReplayMemory
from RL_framework.common.gymenv import GymEnv
from RL_framework.common.model_based import DynamicsModel, MPC

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'CEM'
num_episodes = 10
gamma = 0.9
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 32,
          'training_epoch': 1,
          'control_horizon': 20,
          'K_actions_sample': 50,
          'Best_K': 10,
          'grad': False
          }
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'model_layers': [nn.Linear(env.obs_size + env.action_size, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, env.obs_size)],
                  'reward_layers': [nn.Linear(env.obs_size + env.action_size, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1)]
                  }
learning_rates = dict(model_lr=1e-3, reward_lr=1e-3)
model_loss_fnc = torch.nn.MSELoss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
buffer = ReplayMemory(params['buffer_size'])

model_net = SequentialNetwork(network_layers['model_layers'])
model_opt = optim.Adam(model_net.parameters(), lr=learning_rates['model_lr'])
reward_net = SequentialNetwork(network_layers['reward_layers'])
rew_opt = optim.Adam(reward_net.parameters(), lr=learning_rates['reward_lr'])
dynamics = DynamicsModel(model_net, buffer, model_loss_fnc, model_opt, env, 'diff', reward_net, rew_opt)
MPC = MPC(model_net, env, gamma, reward_net)

# Gather data and training
# ~~~~~~~~~~~~~~~~~~~~~~~~
global_step = buffer.populate_randomly(env, 0.1)
dynamics.train_model(20, params['minibatch_size'], noise_std=0.001)
dynamics.train_reward_fnc(20, params['minibatch_size'])

for episode in tqdm(range(num_episodes)):
    episode_step = 0
    episode_reward = 0
    state = env.env.reset()
    terminal = False
    while terminal is False:
        action = MPC.cem_planning(state, params['K_actions_sample'], params['control_horizon'], params['Best_K'],
                                  episode_step, keep_best_k=False, alpha=0, grad=params['grad'])
        action_scaled = (env.action_low + (env.action_high - env.action_low) * (action + 1) / 2).numpy()
        next_state, reward, terminal, _ = env.env.step(action_scaled)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_reward += reward
        episode_step += 1
        global_step += 1
        buffer.add(state, action, reward, next_state, terminal, None, None)
        state = next_state

        if (episode_step % params['sample_collection'] == 0 or terminal is True) and\
                len(buffer) >= params['minibatch_size']:

            # Train dynamics model
            # ~~~~~~~~~~~~~~~~~~~~
            dynamics.train_model(params['training_epoch'], params['minibatch_size'], noise_std=0.001)
            dynamics.train_reward_fnc(params['training_epoch'], params['minibatch_size'])

    wandb.log({"episode_reward": episode_reward, 'episode': episode})
