from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.networks import SequentialNetwork, ValueFunction, PolicyFunction
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
wandb.config.algorithm = 'AC'
num_episodes = 5

gamma = 0.99
params = {'sample_collection': 1,  # AC=1, PPO=32
          'buffer_size': 1,  # AC=1, PPO=32
          'minibatch_size': 1,  # AC=1, PPO=32
          'actor_grad_steps': 1,  # AC=1, PPO=80
          'critic_grad_steps': 1,  # AC=1, PPO=80
          }
clip_ratio = 0.2  # for PPO

wandb.config.gamma = gamma
wandb.config.update(params)
wandb.config.clip_ratio = clip_ratio

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'policy_layers': [nn.Linear(obs_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, n_actions),
                                    nn.Softmax(dim=0)],
                  'value_layers': [nn.Linear(obs_size, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1)]
                  }
wandb.config.update(network_layers)

learning_rates = dict(policy_lr=3e-4, value_lr=1e-3)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net = SequentialNetwork(network_layers['value_layers'])
actor = PolicyFunction(policy_net, optim.Adam, learning_rates['policy_lr'])
critic = ValueFunction(value_net, optim.Adam, learning_rates['value_lr'], torch.nn.SmoothL1Loss())

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
            transitions = ProcessMinibatch(minibatch, gamma)

            # actor critic
            discounted_gamma = transitions.discount_gamma()
            td_error = transitions.td_error(critic.net).detach()
            log_prob = transitions.log_prob(actor.net)
            loss = -(discounted_gamma * td_error * log_prob).mean()
            actor.optimise(loss)

            target = transitions.one_step_target(critic.net)
            current_v = transitions.current_value(critic.net)
            critic.optimise(target, current_v)
            '''

            # PPO
            td_errors = transitions.td_error(critic.net).detach()
            advantage = transitions.discounted_cumsum(td_errors)
            advantage = (advantage - advantage.mean()) / advantage.std()
            old_action_probs = transitions.action_log_prob.detach()

            for i in range(params['actor_grad_steps']):
                action_prob = transitions.log_prob(actor.net)
                ratio = torch.exp(action_prob - old_action_probs)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                loss = -(torch.min(ratio, clip_adv) * advantage).mean()
                actor.optimise(loss)

            rewards_to_go = transitions.discounted_cumsum(transitions.rewards)
            #rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / rewards_to_go.std()
            for i in range(params['critic_grad_steps']):
                current_v = transitions.current_value(critic.net)
                critic.optimise(rewards_to_go, current_v)

            buffer.empty()
            '''

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
