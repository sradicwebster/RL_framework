from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
import gym
from RL_framework.networks import SequentialNetwork, Qnet_continuous_actions, ValueFunction, PolicyFunction
from RL_framework.buffer import ReplayMemory, ProcessMinibatch
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
wandb.config.algorithm = 'DDPG'
num_episodes = 500

gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 10000,
          'minibatch_size': 64}
action_noise = normal.Normal(0, 0.1)  # add OU noise?
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
policy_layers = [nn.Linear(obs_size, 32),
                 nn.ReLU(),
                 nn.Linear(32, 64),
                 nn.ReLU(),
                 nn.Linear(64, action_n),
                 nn.Hardtanh(action_low, action_high)]
learning_rates = dict(policy_lr=5e-4, value_lr=1e-3)
critic_loss_fnc = torch.nn.SmoothL1Loss()
tau = 0.001

wandb.config.policy_layers = policy_layers
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = SequentialNetwork(policy_layers)
value_net = Qnet_continuous_actions(obs_size, action_n)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt = optim.Adam(value_net.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic = ValueFunction(value_net, value_opt, target_net=True, tau=tau)

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

        action = [(actor.get_policy(state) + action_noise.sample()).item()]

        next_state, reward, terminal, _ = env.step(action)
        step += 1

        buffer.add(state, action, reward, next_state, terminal, step, None)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and len(buffer) >= params['minibatch_size']:
            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)  # t = transitions

            target = t.rewards + gamma * (1-t.terminals) * critic.target_net(t.next_states,
                                                                             actor.target_net(t.next_states)).detach()
            current_v = critic.net(t.states, t.actions)
            critic_loss = critic_loss_fnc(target, current_v)
            wandb.log({"value_loss": critic_loss}, commit=False)
            critic.optimise(critic_loss)

            actor_loss = critic.net(t.states, actor.net(t.states)).mean()
            wandb.log({"policy_loss": actor_loss}, commit=False)
            actor.optimise(-actor_loss)

            critic.soft_target_update()
            actor.soft_target_update()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
