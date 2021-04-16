import torch.nn as nn
import torch.optim as optim
from torch.distributions import normal
import wandb
from tqdm import tqdm
from RL_framework.common.networks import SequentialNetwork, QnetContinuousActions, ValueFunction, PolicyFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.utils import *
from RL_framework.common.gymenv import GymEnv


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'MVE'
num_episodes = 150
gamma = 0.99
params = {'sample_collection': 10,
          'buffer_size': 5000,
          'minibatch_size': 64,
          'training_epoch': 5,
          'sampled_transitions': 64,
          'imagination_steps': 3
          }
action_noise = normal.Normal(0, 0.01)
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'model_layers': [nn.Linear(env.obs_size + env.action_size, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, env.obs_size)],
                  'policy_layers': [nn.Linear(env.obs_size, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, env.action_size),
                                    nn.Tanh()]
                  }
learning_rates = dict(model_lr=1e-4, policy_lr=5e-4, value_lr=1e-3)
loss_fnc = torch.nn.MSELoss()
tau = 0.01
wandb.config.update(network_layers)
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
model_net = SequentialNetwork(network_layers['model_layers'])
model_opt = optim.Adam(model_net.parameters(), lr=learning_rates['model_lr'])

policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net = QnetContinuousActions(env.obs_size, env.action_size)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt = optim.Adam(value_net.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic = ValueFunction(value_net, value_opt, target_net=True, tau=tau)

buffer = ReplayMemory(params['buffer_size'])

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
            for i in range(params['training_epoch']):
                minibatch = buffer.random_sample(params['minibatch_size'])
                t = ProcessMinibatch(minibatch)
                target = t.next_states + torch.normal(0, 0.1, size=t.states.shape)
                state_actions = torch.cat((t.states, t.actions), dim=1)
                current = model_net(state_actions + torch.normal(0, 0.1, size=state_actions.shape))
                loss = loss_fnc(target, current)
                wandb.log({"model_loss": loss}, commit=False)
                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

            # Train value and policy networks
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i in range(params['sampled_transitions']):
                term = True
                while term is True:
                    minibatch = buffer.random_sample(1)
                    t = ProcessMinibatch(minibatch)
                    term = bool(t.terminals)

                actor_loss = critic.net(t.states, actor.net(t.states))
                wandb.log({"policy_loss": actor_loss}, commit=False)
                actor.optimise(-actor_loss)

                imagine_state = t.next_states[0]
                for j in range(params['imagination_steps']-1):
                    with torch.no_grad():
                        imagine_action = actor.target_net(imagine_state)
                        t.states = torch.cat((t.states, imagine_state.reshape(1, -1)))
                        t.actions = torch.cat((t.actions, imagine_action.reshape(1, -1)))
                        reward = env.reward_func(imagine_state.numpy(), imagine_action.numpy())
                        t.rewards = torch.cat((t.rewards, torch.Tensor([gamma ** j * reward]).reshape(1, -1)))
                        imagine_next_state = model_net(torch.cat((imagine_state, imagine_action)))
                        imagine_state = imagine_next_state

                with torch.no_grad():
                    bootstrap_Q = gamma ** params['imagination_steps'] *\
                                  critic.target_net(imagine_state.reshape(1, -1),
                                                    actor.target_net(imagine_state).reshape(1, -1))
                target = torch.stack([t.rewards[i:].sum() + bootstrap_Q for i in range(len(t.rewards))]).reshape(-1, 1)
                current = critic.net(t.states, t.actions)
                critic_loss = loss_fnc(target, current)
                wandb.log({"value_loss": critic_loss}, commit=False)
                critic.optimise(critic_loss)

                critic.soft_target_update()
                actor.soft_target_update()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)

