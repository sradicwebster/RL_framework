from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.common.networks import SequentialNetwork, ValueFunction, PolicyFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.utils import *
import wandb


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'PPO'
num_episodes = 500

gamma = 0.99
params = {'sample_collection': 32,
          'buffer_size': 32,
          'minibatch_size': 32,
          'actor_grad_steps': 5,
          'critic_grad_steps': 20}
clip_ratio = 0.1

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
                                    nn.Softmax(dim=1)],
                  'value_layers': [nn.Linear(obs_size, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1)]
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

            td_error = (t.rewards + gamma * (1 - t.terminals) * critic.net(t.next_states) - critic.net(t.states))\
                .detach()
            discounted_gamma = gamma ** t.steps
            advantage = discounted_cumsum(td_error, discounted_gamma)
            # advantage = (advantage - advantage.mean()) / advantage.std()
            old_action_probs = t.action_log_prob.reshape(-1, 1).detach()

            for grad_step in range(params['actor_grad_steps']):
                action_prob = torch.gather(actor.net(t.states), 1, t.actions).log()
                ratio = torch.exp(action_prob - old_action_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                actor_loss = (torch.min(ratio, clipped_ratio) * advantage).mean()
                wandb.log({"policy_loss": actor_loss}, commit=False)
                actor.optimise(-actor_loss)

            rewards_to_go = discounted_cumsum(t.rewards, discounted_gamma)
            for grad_step in range(params['critic_grad_steps']):
                current_v = critic.net(t.states)
                critic_loss = critic_loss_fnc(rewards_to_go, current_v)
                wandb.log({"value_loss": critic_loss}, commit=False)
                critic.optimise(critic_loss)

            buffer.empty()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
