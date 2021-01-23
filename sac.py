from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
import gym
from RL_framework.common.networks import Qnet_continuous_actions, Squashed_Gaussian, ValueFunction, SACPolicy
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
import wandb

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('Pendulum-v0')
obs_size = env.observation_space.shape[0]
#  action space currently only for 1 action
action_high = env.action_space.high.item()
action_low = env.action_space.low.item()
action_n = 1

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'SAC'
num_episodes = 500

gamma = 0.99
alpha = 0.2  # fixed entropy regularization coefficient
params = {'sample_collection': 1,
          'buffer_size': 10000,
          'minibatch_size': 64}

wandb.config.gamma = gamma
wandb.config.alpha = alpha
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
learning_rates = dict(policy_lr=5e-4, value_lr=1e-3)
critic_loss_fnc = torch.nn.SmoothL1Loss()
tau = 0.005
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = Squashed_Gaussian(obs_size, action_n)
value_net1 = Qnet_continuous_actions(obs_size, action_n)
value_net2 = Qnet_continuous_actions(obs_size, action_n)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt1 = optim.Adam(value_net1.parameters(), lr=learning_rates['value_lr'])
value_opt2 = optim.Adam(value_net2.parameters(), lr=learning_rates['value_lr'])
actor = SACPolicy(policy_net, policy_opt)
critic1 = ValueFunction(value_net1, value_opt1, target_net=True, tau=tau)
critic2 = ValueFunction(value_net2, value_opt2, target_net=True, tau=tau)

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

        action, action_log_prob = actor.action_selection(torch.from_numpy(state).float().reshape(1, -1))

        next_state, reward, terminal, _ = env.step([torch.clamp(action, action_low, action_high).item()])
        step += 1

        buffer.add(state, action, reward, next_state, terminal, step, action_log_prob)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and len(buffer) >= params['minibatch_size']:
            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)  # t = transitions

            target_action, target_action_log = actor.action_selection(t.next_states)
            target = t.rewards + gamma * (1 - t.terminals) * (torch.min(critic1.target_net(t.next_states, target_action),
                                                                       critic2.target_net(t.next_states, target_action))
                                                              - alpha * target_action_log.reshape(-1, 1)).detach()
            current_v1 = critic1.net(t.states, t.actions)
            current_v2 = critic2.net(t.states, t.actions)
            critic_loss1 = critic_loss_fnc(target, current_v1)
            critic_loss2 = critic_loss_fnc(target, current_v2)
            wandb.log({"value_loss": (critic_loss1+critic_loss2)/2}, commit=False)
            critic1.optimise(critic_loss1)
            critic2.optimise(critic_loss2)
            critic1.soft_target_update()
            critic2.soft_target_update()

            policy_action, policy_action_log = actor.action_selection(t.states)
            Q_min = torch.min(critic1.net(t.states, policy_action),
                                                                       critic2.net(t.states, policy_action))
            actor_loss = (Q_min - alpha * policy_action_log.reshape(-1, 1)).mean()
            wandb.log({"policy_loss": actor_loss}, commit=False)
            actor.optimise(-actor_loss)

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
