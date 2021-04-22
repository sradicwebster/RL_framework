import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
import wandb
from tqdm import tqdm
from RL_framework.common.networks import SequentialNetwork, QnetContinuousActions, ValueFunction, PolicyFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.utils import *
from RL_framework.common.gymenv import GymEnv
from RL_framework.common.model_based import DynamicsModel

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'MVE'
num_episodes = 50
gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 128,
          'training_epoch': 20,
          'sampled_transitions': 64,
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
learning_rates = dict(model_lr=1e-3, policy_lr=1e-4, value_lr=5e-4)
model_loss_fnc = torch.nn.MSELoss()
critic_loss_fnc = torch.nn.SmoothL1Loss()
tau = 0.001
wandb.config.update(network_layers)
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
buffer = ReplayMemory(params['buffer_size'])

model_net = SequentialNetwork(network_layers['model_layers'])
model_opt = optim.Adam(model_net.parameters(), lr=learning_rates['model_lr'])
dynamics = DynamicsModel(model_net, buffer, model_loss_fnc, model_opt, env, gamma)

policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net = QnetContinuousActions(env.obs_size, env.action_size)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt = optim.Adam(value_net.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic = ValueFunction(value_net, value_opt, target_net=True, tau=tau)

# Gather data and train dynamics model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            dynamics.train_model(params['training_epoch'], params['minibatch_size'])

            # Train value and policy networks
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i in range(params['sampled_transitions']):

                minibatch = buffer.random_sample(1)
                t = ProcessMinibatch(minibatch)

                actor_loss = critic.net(t.states, actor.net(t.states)).mean()
                wandb.log({"policy_loss": actor_loss}, commit=False)
                actor.optimise(-actor_loss)

                imagine_state = t.next_states[0]
                for j in range(params['imagination_steps']):
                    with torch.no_grad():
                        imagine_action = actor.target_net(imagine_state)
                        reward = env.reward_func(imagine_state.numpy(), imagine_action.numpy())
                        imagine_next_state = dynamics.model(torch.cat((imagine_state, imagine_action)))

                        t.states = torch.cat((t.states, imagine_state.reshape(1, -1)))
                        t.actions = torch.cat((t.actions, imagine_action.reshape(1, -1)))
                        t.rewards = torch.cat((t.rewards, torch.Tensor([gamma ** j * reward]).reshape(1, -1)))
                        imagine_state = imagine_next_state

                with torch.no_grad():
                    imagine_action = actor.target_net(imagine_state)
                    bootstrap_Q = gamma ** params['imagination_steps'] * \
                        critic.target_net(imagine_state.reshape(1, -1), imagine_action.reshape(1, -1))
                target = torch.stack([t.rewards[i:].sum() + bootstrap_Q for i in range(len(t.rewards))]).reshape(-1, 1)
                current = critic.net(t.states, t.actions)
                critic_loss = critic_loss_fnc(target, current)
                #critic_loss = (target - current)**2 / params['imagination_steps']
                wandb.log({"value_loss": critic_loss}, commit=False)
                critic.optimise(critic_loss)

                critic.soft_target_update()
                actor.soft_target_update()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
