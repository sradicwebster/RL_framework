from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from RL_framework.common.networks import SequentialNetwork
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.utils import *
from RL_framework.common.model_based import MPC
from RL_framework.common.gymenv import GymEnv
import wandb


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('CartPole-v0')  # must be discrete action space for mbmf
env.obs_high[1] = 4  # replace Int for env observation max
env.obs_high[3] = 5

# General details
# ~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'MBMF'
num_episodes = 200

gamma = 0.99
params = {'sample_collection': 5,
          'buffer_size': 2000,
          'minibatch_size': 64,
          'random_buffer_size': 2000,
          'training_epoch': 10,
          'control_horizon': 10,
          'K_actions_sample': 50,
          'dataset_ratio': 0.9
          }
wandb.config.gamma = gamma
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'model_layers': [nn.Linear(env.obs_size + 1, 32),  # +1 for discrete action space
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, env.obs_size)]
                  }
learning_rates = dict(model_lr=1E-3)
loss_fnc = torch.nn.MSELoss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
model_net = SequentialNetwork(network_layers['model_layers'])
opt = optim.Adam(model_net.parameters(), lr=learning_rates['model_lr'])
dataset_random = ReplayMemory(params['random_buffer_size'])
dataset_rl = ReplayMemory(params['buffer_size'])
MPC = MPC(model_net, env, gamma)

# Gather random data and train dynamics models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while len(dataset_random) < params['random_buffer_size']:
    state = env.env.reset() + torch.normal(0, 0.001, size=(env.obs_size,)).numpy()
    terminal = False
    while terminal is False:
        action = torch.randint(env.action_size, size=(1,)).item()
        next_state, reward, terminal, _ = env.env.step(action)
        dataset_random.add(state, action, reward, next_state, terminal, None, None)
        state = next_state

for i in range(params['training_epoch']):
    minibatch = dataset_random.random_sample(params['minibatch_size'])
    t = ProcessMinibatch(minibatch)
    t.standardise(env.obs_high)
    target = t.next_states - t.states + torch.normal(0, 0.001, size=t.states.shape)
    state_actions = torch.cat((t.states, t.actions), dim=1)
    current = model_net(state_actions + torch.normal(0, 0.001, size=state_actions.shape))
    loss = loss_fnc(target, current)
    wandb.log({"model_loss": loss})
    opt.zero_grad()
    loss.backward()
    opt.step()

# Model based controller loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
global_step = 0
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    episode_step = 0
    state = env.env.reset()
    terminal = False
    while terminal is False:
        action = MPC.random_shooting(state, params['K_actions_sample'], params['control_horizon'])
        next_state, reward, terminal, _ = env.env.step(action)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_step += 1
        global_step += 1
        dataset_rl.add(state, action, reward, next_state, terminal, episode_step, None)
        state = next_state
        episode_reward += reward

        if (episode_step % params['sample_collection'] == 0 or terminal is True) and \
                len(dataset_rl) >= params['minibatch_size']:

            minibatch_random = dataset_random.random_sample(
                round(params['minibatch_size'] * (1 - params['dataset_ratio'])))
            minibatch_rl = dataset_rl.random_sample(round(params['minibatch_size'] * params['dataset_ratio']))
            t_random = ProcessMinibatch(minibatch_random)
            t_rl = ProcessMinibatch(minibatch_rl)
            t_random.standardise(env.obs_high)
            t_rl.standardise(env.obs_high)
            target = torch.cat((t_random.next_states, t_rl.next_states)) - torch.cat((t_random.states, t_rl.states))
            state_actions = torch.cat((torch.cat((t_random.states, t_rl.states)),
                                       torch.cat((t_random.actions, t_rl.actions))), dim=1)
            current = model_net(state_actions)
            loss = loss_fnc(target, current)
            wandb.log({"model_loss": loss, 'step': global_step, 'episode': episode}, commit=False)
            opt.zero_grad()
            loss.backward()
            opt.step()

    wandb.log({"episode_reward": episode_reward, "episode": episode})
