from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from RL_framework.common.networks import SequentialNetwork, ValueFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.gymenv import GymEnv
import wandb


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('CartPole-v0')
env.obs_high[1] = 4  # replace Int for env observation max
env.obs_high[3] = 5

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'DQN'
num_episodes = 1000

gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 32,
          'target_steps_update': 1000}
epsilon = {'eps_start': 0.9,
           'eps_end': 0.1,
           'eps_decay': 200}
wandb.config.gamma = gamma
wandb.config.update(params)
wandb.config.update(epsilon)

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'Qnet_layers': [nn.Linear(env.obs_size, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, env.action_size)],
                  'dueling_layers': [env.obs_size, 64, 128, env.action_size]}
learning_rates = {'Qnet_lr': 2e-4}
loss_function = torch.nn.SmoothL1Loss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
net = SequentialNetwork(network_layers['Qnet_layers'])
wandb.config.value_layers = net.layers
opt = optim.Adam(net.parameters(), lr=learning_rates['Qnet_lr'])
Qnet = ValueFunction(net, opt, epsilon=epsilon, target_net=True)
buffer = ReplayMemory(params['buffer_size'])

# Get training
# ~~~~~~~~~~~~
global_step = 0
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    episode_step = 0
    state = env.env.reset()
    terminal = False
    while terminal is False:
        action = Qnet.epsilon_greedy_action(state, episode)
        next_state, reward, terminal, _ = env.env.step(action)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_step += 1
        global_step += 1
        buffer.add(state, action, reward, next_state, terminal, episode_step, None)
        state = next_state
        episode_reward += reward

        if (episode_step % params['sample_collection'] == 0 or terminal is True) and\
                len(buffer) >= params['minibatch_size']:

            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)

            with torch.no_grad():
                target = t.rewards + (1-t.terminals) * gamma * Qnet.target_net(t.next_states).max(dim=1).values\
                    .reshape(-1, 1)
            current_v = torch.gather(Qnet.net(t.states), 1, t.actions)
            loss = loss_function(target, current_v)
            wandb.log({"value_loss": loss, 'step': global_step, 'episode': episode}, commit=False)
            Qnet.optimise(loss)

        if global_step % params['target_steps_update'] == 0:
            Qnet.hard_target_update()

    wandb.log({"episode_reward": episode_reward, "episode": episode})
