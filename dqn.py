from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.networks import SequentialNetwork, ValueFunction
from RL_framework.buffer import ReplayMemory, ProcessMinibatch
import wandb


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'DQN'
num_episodes = 500

gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 10000,
          'minibatch_size': 32,
          'target_steps_update': 2000}
epsilon = {'eps_start': 1,
           'eps_end': 0.1,
           'eps_decay': 250}
wandb.config.gamma = gamma
wandb.config.update(params)
wandb.config.update(epsilon)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'Qnet_layers': [nn.Linear(obs_size, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, n_actions)],
                  'dueling_layers': [obs_size, 64, 128, n_actions]}
learning_rates = {'Qnet_lr': 2e-4}
loss_function = torch.nn.SmoothL1Loss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
net = SequentialNetwork(network_layers['Qnet_layers'])
wandb.config.value_layers = net.layers
opt = optim.Adam(net.parameters(), lr=learning_rates['Qnet_lr'])
Qnet = ValueFunction(net, opt, epsilon=epsilon, target_net=True)

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
            t = ProcessMinibatch(minibatch)

            target = t.rewards + (1-t.terminals) * gamma * torch.max(Qnet.target_net(t.next_states), dim=1).values\
                .reshape(-1, 1)
            current_v = torch.gather(Qnet.net(t.states), 1, t.actions)
            loss = loss_function(target, current_v)
            wandb.log({"value_loss": loss}, commit=False)
            Qnet.optimise(loss)

        if total_step % params['target_steps_update'] == 0:
            Qnet.hard_target_update()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
