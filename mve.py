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
wandb.config.algorithm = 'MVE'
num_episodes = 200
gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 32,
          'training_epoch': 1,
          'sampled_transitions': 32,
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
critic_loss_fnc = torch.nn.MSELoss()
tau = 0.005
wandb.config.update(network_layers)
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
buffer = ReplayMemory(params['buffer_size'])

model_net = SequentialNetwork(network_layers['model_layers'])
model_opt = optim.Adam(model_net.parameters(), lr=learning_rates['model_lr'])
dynamics = DynamicsModel(model_net, buffer, model_loss_fnc, model_opt, env, type='diff')

policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net = QnetContinuousActions(env)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt = optim.Adam(value_net.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic = ValueFunction(value_net, value_opt, target_net=True, tau=tau)

# Gather data and training
# ~~~~~~~~~~~~~~~~~~~~~~~~
global_step = buffer.populate_randomly(env, 0.1)
dynamics.train_model(20, params['minibatch_size'], noise_std=0.001)
buffer.empty()

for episode in tqdm(range(num_episodes)):
    episode_step = 0
    episode_reward = 0
    state = env.env.reset()
    terminal = False
    while terminal is False:
        with torch.no_grad():
            action = torch.clamp(actor.get_policy(state) + action_noise.sample(env.action_high.shape), -1, 1)
        action_scaled = env.action_low + (env.action_high - env.action_low) * (action + 1) / 2
        next_state, reward, terminal, _ = env.env.step(action_scaled)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_reward += reward
        episode_step += 1
        global_step += 1
        buffer.add(state, action, reward, next_state, terminal, None, None)
        state = next_state

        if (episode_step % params['sample_collection'] == 0 or terminal is True) and\
                len(buffer) >= params['minibatch_size']:

            # Train dynamics model
            # ~~~~~~~~~~~~~~~~~~~~
            dynamics.train_model(params['training_epoch'], params['minibatch_size'], noise_std=0.001)

            # Train value and policy networks
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for _ in range(params['sampled_transitions']):

                minibatch = buffer.random_sample(1)
                t = ProcessMinibatch(minibatch)

                actor_loss = critic.net(t.states, actor.net(t.states))
                wandb.log({"policy_loss": actor_loss, 'step': global_step, 'episode': episode}, commit=False)
                actor.optimise(-actor_loss)

                imagine_state = t.next_states
                with torch.no_grad():
                    for j in range(params['imagination_steps']):
                        imagine_action = actor.target_net(imagine_state)
                        imagine_action_scaled = env.action_low + (env.action_high - env.action_low) * (imagine_action
                                                                                                       + 1) / 2
                        reward = env.reward_func(imagine_state.squeeze().numpy(),
                                                 imagine_action_scaled.squeeze().numpy())
                        imagine_next_state = imagine_state + dynamics.model(torch.cat((imagine_state, imagine_action),
                                                                                      dim=1))

                        t.states = torch.cat((t.states, imagine_state))
                        t.actions = torch.cat((t.actions, imagine_action))
                        t.rewards = torch.cat((t.rewards, torch.Tensor([gamma ** (j + 1) * reward]).reshape(1, -1)))
                        imagine_state = imagine_next_state

                    imagine_action = actor.target_net(imagine_state).reshape(1, -1)
                    bootstrap_Q = gamma ** (params['imagination_steps'] + 1) * critic.target_net(imagine_state,
                                                                                                 imagine_action)

                target = torch.stack([t.rewards[i:].sum() + bootstrap_Q for i in range(len(t.rewards))]).reshape(-1, 1)
                current = critic.net(t.states, t.actions)
                critic_loss = critic_loss_fnc(target, current)
                wandb.log({"value_loss": critic_loss, 'step': global_step, 'episode': episode}, commit=False)
                critic.optimise(critic_loss)

                critic.soft_target_update()
                actor.soft_target_update()

    wandb.log({"episode_reward": episode_reward, 'episode': episode})
