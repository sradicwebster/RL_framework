from tqdm import tqdm
import wandb
import torch
import torch.optim as optim
from torch.distributions import normal
from RL_framework.common.networks import DeterministicPolicy, QnetContinuousActions, ValueFunction, PolicyFunction
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.gymenv import GymEnv


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'DDPG'
num_episodes = 200
gamma = 0.99
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 32}
action_noise = normal.Normal(0, 0.05)
wandb.config.gamma = gamma
wandb.config.update(params)
wandb.config.action_noise = action_noise

# Networks details
# ~~~~~~~~~~~~~~~~
learning_rates = dict(policy_lr=5e-4, value_lr=1e-3)
critic_loss_fnc = torch.nn.MSELoss()
tau = 0.001
wandb.config.update(learning_rates)
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = DeterministicPolicy(env)
value_net = QnetContinuousActions(env)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt = optim.Adam(value_net.parameters(), lr=learning_rates['value_lr'], weight_decay=1e-2)
actor = PolicyFunction(policy_net, policy_opt, target_net=True, tau=tau)
critic = ValueFunction(value_net, value_opt, target_net=True, tau=tau)
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
        with torch.no_grad():
            action = actor.get_policy(state)
        action = torch.stack([torch.clamp(action[i] + action_noise.sample(), env.action_low[i], env.action_high[i])
                              for i in range(env.action_size)]).numpy()
        next_state, reward, terminal, _ = env.env.step(action)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_reward += reward
        episode_step += 1
        global_step += 1
        buffer.add(state, action, reward, next_state, terminal, episode_step, None)
        state = next_state

        if (episode_step % params['sample_collection'] == 0 or terminal is True) \
                and len(buffer) >= params['minibatch_size']:
            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)

            with torch.no_grad():
                target_action = actor.target_net(t.next_states)
                target = t.rewards + gamma * (1-t.terminals) * critic.target_net(t.next_states, target_action)
            current_v = critic.net(t.states, t.actions)
            critic_loss = critic_loss_fnc(target, current_v)
            wandb.log({"value_loss": critic_loss,  'step': global_step, 'episode': episode}, commit=False)
            critic.optimise(critic_loss)

            actor_loss = critic.net(t.states, actor.net(t.states)).mean()
            wandb.log({"policy_loss": actor_loss,  'step': global_step, 'episode': episode}, commit=False)
            actor.optimise(-actor_loss)

            critic.soft_target_update()
            actor.soft_target_update()

    wandb.log({"episode_reward": episode_reward, 'episode': episode})
