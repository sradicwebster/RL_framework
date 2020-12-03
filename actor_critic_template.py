from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from networks import SequentialNetwork, PolicyFunction, ValueFunction
from buffer import ReplayMemory
from updates import ProcessMinibatch
import wandb

wandb.init(project='framework')
wandb.config = wandb.config

# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# General details
# ~~~~~~~~~~~~~~~
wandb.config.algorithm = 'AC'
gamma = 0.99
sample_collection = 1  # AC=1, DQN=1, PPO=32
buffer_size = 1  # AC=1, DQN=1000, PPO=32
minibatch_size = 1  # AC=1, DQN=32, PPO=32

num_episodes = 500

wandb.config.gamma = gamma
wandb.config.sample_collection = sample_collection
wandb.config.buffer_size = buffer_size
wandb.config.minibatch_size = minibatch_size

# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'policy_layers': [nn.Linear(obs_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, n_actions),
                                    nn.Softmax(dim=0)],
                  'value_layers': [nn.Linear(obs_size, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1)]
                  }
wandb.config.update(network_layers)

learning_rates = dict(policy_lr=1e-4, value_lr=5e-3)
wandb.config.update(learning_rates)

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = SequentialNetwork(network_layers['policy_layers'])
value_net = SequentialNetwork(network_layers['value_layers'])
actor = PolicyFunction(policy_net, optim.Adam, learning_rates['policy_lr'])
critic = ValueFunction(value_net, optim.Adam, learning_rates['value_lr'], torch.nn.MSELoss())

buffer = ReplayMemory(buffer_size)

# Get training
# ~~~~~~~~~~~~
episode_rewards = []
for episode in tqdm(range(num_episodes)):

    episode_reward = 0
    step = 0

    state = env.reset()

    terminal = False
    while terminal is False:

        action = actor.softmax_action(state)

        next_state, reward, terminal, _ = env.step(action)
        step += 1

        buffer.add(state, action, reward, next_state, terminal, step)

        state = next_state
        episode_reward += reward

        if (step % sample_collection == 0 or terminal is True) and len(buffer) >= minibatch_size:
            minibatch = buffer.sample(minibatch_size)
            transitions = ProcessMinibatch(minibatch)

            target = transitions.one_step_target(critic.net, gamma)
            current_v = transitions.current_value(critic.net)
            critic.optimise(target, current_v)

            discount = transitions.discount(gamma)
            log_prob = transitions.log_prob(actor)
            error = torch.flatten(target - current_v).tolist()
            loss = -torch.stack([discount[i] * error[i] * log_prob[i] for i in range(len(discount))]).mean()
            actor.optimise(loss)

        # add buffer.empty() function for PPO (also sample in order)

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)
