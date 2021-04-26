import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
import wandb
from tqdm import tqdm
from RL_framework.common.networks import SequentialNetwork, QnetContinuousActions, ValueFunction, PolicyFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.gymenv import GymEnv
from RL_framework.common.model_based import DynamicsModel

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'STEVE'
num_episodes = 100
gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 64,
          'training_epoch': 5,
          'imagination_steps': 1
          }
action_noise = normal.Normal(0, 0.05)
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
                  'policy_layers': [nn.Linear(env.obs_size, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, env.action_size),
                                    nn.Tanh()]
                  }
learning_rates = dict(model_lr=1e-3, policy_lr=5e-4, value_lr=1e-3)
model_loss_fnc = torch.nn.MSELoss()
critic_loss_fnc = torch.nn.SmoothL1Loss()
tau = 0.001
wandb.config.update(network_layers)
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
buffer = ReplayMemory(params['buffer_size'])

model_net1 = SequentialNetwork(network_layers['model_layers'])
model_net2 = SequentialNetwork(network_layers['model_layers'])
model_opt1 = optim.Adam(model_net1.parameters(), lr=learning_rates['model_lr'])
model_opt2 = optim.Adam(model_net2.parameters(), lr=learning_rates['model_lr'])
dynamics1 = DynamicsModel(model_net1, buffer, model_loss_fnc, model_opt1, env, gamma)
dynamics2 = DynamicsModel(model_net2, buffer, model_loss_fnc, model_opt2, env, gamma)
dynamics = [dynamics1, dynamics2]

policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net1 = QnetContinuousActions(env.obs_size, env.action_size)
value_net2 = QnetContinuousActions(env.obs_size, env.action_size)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt1 = optim.Adam(value_net1.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
value_opt2 = optim.Adam(value_net2.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic1 = ValueFunction(value_net1, value_opt1, target_net=True, tau=tau)
critic2 = ValueFunction(value_net2, value_opt2, target_net=True, tau=tau)
critics = [critic1, critic2]

# Gather data and training
# ~~~~~~~~~~~~~~~~~~~~~~~~
dynamics1.populate_buffer_randomly(0.1)

episode_rewards = []
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    step = 0
    state = env.env.reset()
    terminal = False
    while terminal is False:
        action = actor.get_policy(state)
        action = [torch.clamp(action[i] + action_noise.sample(), env.action_low[i], env.action_high[i]).item()
                  for i in range(env.action_size)]
        next_state, reward, terminal, _ = env.env.step(action)
        step += 1
        buffer.add(state, action, reward, next_state, terminal, None, None)
        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and len(buffer) >= params['minibatch_size']:
            # Train dynamics model
            # ~~~~~~~~~~~~~~~~~~~~
            for dynamic in dynamics:
                dynamic.train_model(params['training_epoch'], params['minibatch_size'])

            # Update policy network
            # ~~~~~~~~~~~~~~~~~~~~~
            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)

            actor_loss = critics[0].net(t.states, actor.net(t.states)).mean()
            wandb.log({"policy_loss": actor_loss}, commit=False)
            actor.optimise(-actor_loss)

            # Update critic networks
            # ~~~~~~~~~~~~~~~~~~~~~~
            targets = torch.zeros((params['minibatch_size'], params['imagination_steps']+1, len(dynamics),
                                   len(critics)))
            with torch.no_grad():
                for c, critic in enumerate(critics):
                    target = t.rewards + gamma * critic.target_net(t.next_states, actor.target_net(t.next_states))
                    targets[:, 0, :, c] = torch.ones(params['minibatch_size'], len(dynamics)) * target

                #imagine_state = [t.next_states] * len(dynamics)
                imagine_state = t.next_states
                for h in range(1, params['imagination_steps']+1):  # only works for H=1
                    imagine_action = actor.target_net(imagine_state)
                    reward = [env.reward_func(imagine_state[i].squeeze().numpy(), imagine_action[i].squeeze().numpy())
                              for i in range(len(imagine_action))]
                    dis_reward = gamma ** h * torch.Tensor(reward).reshape(-1, 1)

                    for d, dynamic in enumerate(dynamics):
                        imagine_next_state = dynamic.model(torch.cat((imagine_state, imagine_action), dim=1))

                        for c, critic in enumerate(critics):
                            target = dis_reward + gamma ** (h+1) \
                                     * critic.target_net(imagine_next_state, actor.target_net(imagine_next_state))
                            targets[:, h, d, c] = target.squeeze()

            target = targets.mean(dim=(1, 2, 3)).reshape(-1, 1)

            critic_losses = []
            for critic in critics:
                current = critic.net(t.states, t.actions)
                critic_loss = critic_loss_fnc(target, current)
                critic_losses.append(critic_loss)
                critic.optimise(critic_loss)
                critic.soft_target_update()
            wandb.log({"value_loss": torch.mean(torch.stack(critic_losses))}, commit=False)

            actor.soft_target_update()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
