from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.networks import SequentialNetwork, ValueFunction, PolicyFunction
from RL_framework.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.utils import *
import wandb


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# General details
# ~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'AC'
num_episodes = 500

gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 1,
          'minibatch_size': 1}
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'policy_layers': [nn.Linear(obs_size, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, n_actions),
                                    nn.Softmax(dim=1)],
                  'value_layers': [nn.Linear(obs_size, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1)]
                  }
learning_rates = dict(policy_lr=1e-4, value_lr=1e-3)
critic_loss_fnc = torch.nn.SmoothL1Loss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net = SequentialNetwork(network_layers['value_layers'])
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt = optim.Adam(value_net.parameters(), lr=learning_rates['value_lr'])
actor = PolicyFunction(policy_net, policy_opt)
critic = ValueFunction(value_net, value_opt)

buffer = ReplayMemory(params['buffer_size'])

# Get training
# ~~~~~~~~~~~~
episode_rewards = []
for episode in tqdm(range(num_episodes)):

    episode_reward = 0
    step = 0

    state = env.reset()

    terminal = False
    while terminal is False:

        action, action_log_prob = actor.softmax_action(state)
        next_state, reward, terminal, _ = env.step(action)

        step += 1

        buffer.add(state, action, reward, next_state, terminal, step, action_log_prob)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and len(buffer) >= params['minibatch_size']:

            minibatch = buffer.ordered_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)

            target = t.rewards + gamma * (1 - t.terminals) * critic.net(t.next_states)
            current_v = critic.net(t.states)
            critic_loss = critic_loss_fnc(target, current_v)
            wandb.log({"value_loss": critic_loss}, commit=False)
            critic.optimise(critic_loss)

            advantage = (target - current_v).detach()
            discounted_gamma = gamma ** t.steps
            log_prob = torch.gather(actor.net(t.states), 1, t.actions).log()
            actor_loss = (discounted_gamma * advantage * log_prob).mean()
            wandb.log({"policy_loss": actor_loss}, commit=False)
            actor.optimise(-actor_loss)

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
