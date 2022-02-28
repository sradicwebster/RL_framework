import torch
import torch.nn as nn
import torch.optim as optim
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
num_episodes = 200
gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 64,
          'training_epoch': 2,
          'imagination_steps': 2,
          'grad_steps': 2,
          'updates': 2
          }
epsilon = 0.05
wandb.config.gamma = gamma
wandb.config.update(params)
wandb.config.epsilon = epsilon

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
learning_rates = dict(model_lr=3e-4, policy_lr=3e-4, value_lr=3e-4)
model_loss_fnc = torch.nn.MSELoss()
critic_loss_fnc = torch.nn.MSELoss()
tau = 0.005
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
dynamics1 = DynamicsModel(model_net1, buffer, model_loss_fnc, model_opt1, env, model_type='diff')
dynamics2 = DynamicsModel(model_net2, buffer, model_loss_fnc, model_opt2, env, model_type='diff')
dynamics = [dynamics1, dynamics2]

policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net1 = QnetContinuousActions(env)
value_net2 = QnetContinuousActions(env)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt1 = optim.Adam(value_net1.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
value_opt2 = optim.Adam(value_net2.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic1 = ValueFunction(value_net1, value_opt1, target_net=True, tau=tau)
critic2 = ValueFunction(value_net2, value_opt2, target_net=True, tau=tau)
critics = [critic1, critic2]

# Gather data and training
# ~~~~~~~~~~~~~~~~~~~~~~~~
global_step = buffer.populate_randomly(env, 0.1)
for dynamic in dynamics:
    dynamic.train_model(20, params['minibatch_size'], noise_std=0.001)

for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    episode_step = 0
    state = env.env.reset()
    terminal = False
    while terminal is False:
        with torch.no_grad():
            action = actor.get_policy(state)
        if torch.rand(1) < epsilon:
            action = torch.clamp(torch.normal(action, torch.Tensor([0.1])), -1, 1)
        action_scaled = (env.action_low + (env.action_high - env.action_low) * (action + 1) / 2).numpy()
        next_state, reward, terminal, _ = env.env.step(action_scaled)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_reward += reward
        episode_step += 1
        global_step += 1
        buffer.add(state, action, reward, next_state, terminal, None, None)
        state = next_state

        if (episode_step % params['sample_collection'] == 0 or terminal is True) and \
                len(buffer) >= params['minibatch_size']:

            for _ in range(params['updates']):

                # Train dynamics model
                # ~~~~~~~~~~~~~~~~~~~~
                for dynamic in dynamics:
                    dynamic.train_model(params['training_epoch'], params['minibatch_size'], params['grad_steps'],
                                        noise_std=0.001)

                minibatch = buffer.random_sample(params['minibatch_size'])
                t = ProcessMinibatch(minibatch)

                # Update policy network
                # ~~~~~~~~~~~~~~~~~~~~~
                actor_loss = critics[torch.randint(2, (1,)).item()].net(t.states, actor.net(t.states)).mean()
                wandb.log({"policy_loss": actor_loss, 'step': global_step, 'episode': episode}, commit=False)
                actor.optimise(-actor_loss)

                # Update critic networks
                # ~~~~~~~~~~~~~~~~~~~~~~
                targets = torch.zeros((params['minibatch_size'], params['imagination_steps']+1, len(dynamics),
                                       len(critics)))
                with torch.no_grad():
                    for c, critic in enumerate(critics):
                        target = t.rewards + gamma * critic.target_net(t.next_states, actor.target_net(t.next_states))
                        targets[:, 0, :, c] = torch.ones(params['minibatch_size'], len(dynamics)) * target

                    for d, dynamic in enumerate(dynamics):
                        imagine_state = t.next_states
                        dis_reward = 0
                        for h in range(1, params['imagination_steps']+1):

                            imagine_action = actor.target_net(imagine_state)
                            imagine_action_scaled = env.action_low + (env.action_high - env.action_low) * (imagine_action
                                                                                                           + 1) / 2
                            reward = [env.reward_func(imagine_state[i].squeeze().numpy(), imagine_action_scaled[i].squeeze()
                                                      .numpy()) for i in range(len(imagine_action_scaled))]
                            dis_reward += gamma ** h * torch.Tensor(reward).reshape(-1, 1)

                            imagine_next_state = imagine_state + dynamic.model(torch.cat((imagine_state, imagine_action),
                                                                                         dim=1))
                            imagine_next_state = torch.stack([torch.clamp(imagine_next_state[:, i], env.obs_low[i],
                                                                          env.obs_high[i]) for i in range(env.obs_size)]).T

                            for c, critic in enumerate(critics):
                                target = dis_reward + gamma ** (h+1) \
                                         * critic.target_net(imagine_next_state, actor.target_net(imagine_next_state))
                                targets[:, h, d, c] = target.squeeze()

                            imagine_state = imagine_next_state

                inverse_var = 1 / (targets.var(dim=(2, 3)) + 1e-10)
                normalised_weights = inverse_var / inverse_var.sum(dim=1, keepdims=True)
                model_usage = normalised_weights[:, 0].mean()
                wandb.log({"model_usage": model_usage, 'step': global_step, 'episode': episode}, commit=False)
                target = (targets.mean(dim=(2, 3)) * normalised_weights).sum(dim=1, keepdims=True)

                critic_losses = []
                for critic in critics:
                    current = critic.net(t.states, t.actions)
                    critic_loss = critic_loss_fnc(target, current)
                    critic_losses.append(critic_loss)
                    critic.optimise(critic_loss)
                    critic.soft_target_update()
                wandb.log({"value_loss": torch.stack(critic_losses).mean(), 'step': global_step, 'episode': episode},
                          commit=False)

                actor.soft_target_update()

    wandb.log({"episode_reward": episode_reward, "episode": episode})
