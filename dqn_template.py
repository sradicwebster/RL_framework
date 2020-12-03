from tqdm import tqdm
import torch
import torch.nn as nn
import gym
from networks import *
from buffer import *
from updates import *
import wandb

wandb.init(project='framework')
wandb.config = wandb.config

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# General details
# ~~~~~~~~~~~~~~~
wandb.config.algorithm = 'DQN'
gamma = 0.99
sample_collection = 1  # AC=1, DQN=1, PPO=32
buffer_size = 5000     # AC=1, DQN=1000, PPO=32
minibatch_size = 32    # AC=1, DQN=32, PPO=32

target_update = 500 # steps

wandb.config.gamma = gamma
wandb.config.sample_collection = sample_collection
wandb.config.buffer_size = buffer_size
wandb.config.minibatch_size = minibatch_size

num_episodes = 1000

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'Qnet_layers': [nn.Linear(obs_size, 64),
                           nn.ReLU(),
                           nn.Linear(64, 128),
                           nn.ReLU(),
                           nn.Linear(128, n_actions)]
                 }
wandb.config.update(network_layers)

learning_rates = {'Qnet_lr': 5e-4,}
wandb.config.update(learning_rates)

epsilon = {'eps_start': 1,
          'eps_end': 0.1,
          'eps_decay': 250 ,
          }
wandb.config.update(epsilon)

# Initialisation
# ~~~~~~~~~~~~~~
net = SequentialNetwork(network_layers['Qnet_layers'])
Qnet = ValueFunction(net, optim.Adam, learning_rates['Qnet_lr'], torch.nn.SmoothL1Loss(), epsilon, target_net=True)

buffer = ReplayMemory(buffer_size)

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

        buffer.add(state, action, reward, next_state, terminal, step)

        state = next_state
        episode_reward += reward

        if (step % sample_collection == 0 or terminal == True) and len(buffer) >= minibatch_size:           

            minibatch = buffer.sample(minibatch_size) # may need to change to random_sample
            transitions = ProcessMinibatch(minibatch)
            
            target = transitions.qlearning_target(Qnet.target_net, gamma) 
            current_v = transitions.current_value(Qnet.net)
            Qnet.optimise(target, current_v)
            
        if total_step % target_update == 0:
            Qnet.update_target()
            

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
