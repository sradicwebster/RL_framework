from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from RL_framework.common.networks import SequentialNetwork, QnetContinuousActions, ValueFunction, PolicyFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.gymenv import GymEnv
import wandb

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum', monitor_gym=True)
wandb.config.algorithm = 'TD3'
num_episodes = 500

gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 10000,
          'minibatch_size': 64}
action_noise = Normal(0, 0.1)  # add OU noise?
target_noise = Normal(0, 0.2)
noise_clip = 0.5
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
policy_layers = [nn.Linear(env.obs_size, 32),
                 nn.ReLU(),
                 nn.Linear(32, 64),
                 nn.ReLU(),
                 nn.Linear(64, env.action_size),
                 nn.Tanh()]


def scale_action(act):
    return env.action_low + (act+1)/2 * (env.action_high - env.action_low)


learning_rates = dict(policy_lr=5e-4, value_lr=1e-3)
critic_loss_fnc = torch.nn.SmoothL1Loss()
tau = 0.001
policy_delay = 2

wandb.config.policy_layers = policy_layers
wandb.config.update(learning_rates)
wandb.config.tau = tau
wandb.config.policy_delay = policy_delay

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = SequentialNetwork(policy_layers)
value_net1 = QnetContinuousActions(env.obs_size, env.action_size)
value_net2 = QnetContinuousActions(env.obs_size, env.action_size)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt1 = optim.Adam(value_net1.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
value_opt2 = optim.Adam(value_net2.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic1 = ValueFunction(value_net1, value_opt1, target_net=True, tau=tau)
critic2 = ValueFunction(value_net2, value_opt2, target_net=True, tau=tau)

buffer = ReplayMemory(params['buffer_size'])

# Get training
# ~~~~~~~~~~~~
episode_rewards = []
for episode in tqdm(range(num_episodes)):

    # env = gym.make('Pendulum-v0')
    # if episode % 50 == 0:
    #     env = gym.wrappers.Monitor(env, f'./td3_video/{episode}', force=True)

    episode_reward = 0
    step = 0

    state = env.env.reset()

    terminal = False
    while terminal is False:

        with torch.no_grad():
            action = scale_action(actor.get_policy(state)) + action_noise.sample([env.action_size])
        action = torch.stack([torch.clamp(action[i], env.action_low[i].item(), env.action_high[i].item())
                              for i in range(env.action_size)]).numpy()

        next_state, reward, terminal, _ = env.env.step(action)
        step += 1

        buffer.add(state, action, reward, next_state, terminal, step, None)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and len(buffer) >= params['minibatch_size']:
            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)  # t = transitions

            with torch.no_grad():
                target_action = scale_action(actor.target_net(t.next_states)) + torch.clamp(target_noise.sample(),
                                                                                            -noise_clip, noise_clip)
                target_action = torch.stack([torch.clamp(target_action[:, i], env.action_low[i].item(),
                                                         env.action_high[i].item()) for i in range(env.action_size)])\
                    .reshape(-1, env.action_size)
                q_target = torch.min(critic1.target_net(t.next_states, target_action),
                                     critic2.target_net(t.next_states, target_action))
                target = t.rewards + gamma * (1 - t.terminals) * q_target

            current_v1 = critic1.net(t.states, t.actions)
            current_v2 = critic2.net(t.states, t.actions)
            critic_loss1 = critic_loss_fnc(target, current_v1)
            critic_loss2 = critic_loss_fnc(target, current_v2)
            wandb.log({"value_loss": (critic_loss1+critic_loss2)/2}, commit=False)
            critic1.optimise(critic_loss1)
            critic2.optimise(critic_loss2)

            critic1.soft_target_update()
            critic2.soft_target_update()

            if step % policy_delay == 0:
                actor_loss = critic1.net(t.states, actor.net(t.states)).mean()
                wandb.log({"policy_loss": actor_loss}, commit=False)
                actor.optimise(-actor_loss)

                actor.soft_target_update()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)

torch.save(actor.net, 'saved_models/pendulum/td3/actor.pth')
