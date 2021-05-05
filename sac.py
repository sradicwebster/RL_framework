from tqdm import tqdm
import wandb
import torch
import torch.optim as optim
from RL_framework.common.networks import QnetContinuousActions, GaussianPolicy, ValueFunction, SACPolicy
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.gymenv import GymEnv


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = GymEnv('Pendulum-v0')

# General details
# ~~~~~~~~~~~~~~~
wandb.init(project='framework_pendulum')
wandb.config.algorithm = 'SAC'
num_episodes = 200
gamma = 0.99
alpha = 0.1  # fixed entropy regularization coefficient
params = {'sample_collection': 1,
          'buffer_size': 5000,
          'minibatch_size': 32}

wandb.config.gamma = gamma
wandb.config.alpha = alpha
wandb.config.update(params)

# Networks details
# ~~~~~~~~~~~~~~~~
learning_rates = dict(policy_lr=5e-4, value_lr=1e-3)
critic_loss_fnc = torch.nn.MSELoss()
tau = 0.001
wandb.config.update(learning_rates)
wandb.config.critic_loss_fnc = critic_loss_fnc
wandb.config.tau = tau

# Initialisation
# ~~~~~~~~~~~~~~
policy_net = GaussianPolicy(env)
value_net1 = QnetContinuousActions(env)
value_net2 = QnetContinuousActions(env)
policy_opt = optim.Adam(policy_net.parameters(), lr=learning_rates['policy_lr'])
value_opt1 = optim.Adam(value_net1.parameters(), lr=learning_rates['value_lr'])
value_opt2 = optim.Adam(value_net2.parameters(), lr=learning_rates['value_lr'])
actor = SACPolicy(policy_net, policy_opt)
critic1 = ValueFunction(value_net1, value_opt1, target_net=True, tau=tau)
critic2 = ValueFunction(value_net2, value_opt2, target_net=True, tau=tau)
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
            action, action_log_prob = actor.action_selection(torch.from_numpy(state).float())
        action = torch.stack([torch.clamp(action[i], env.action_low[i], env.action_high[i])
                              for i in range(env.action_size)]).numpy()
        next_state, reward, terminal, _ = env.env.step(action)
        wandb.log({'reward': reward, 'step': global_step, 'episode': episode})
        episode_reward += reward
        episode_step += 1
        global_step += 1
        buffer.add(state, action, reward, next_state, terminal, episode_step, action_log_prob)
        state = next_state

        if (episode_step % params['sample_collection'] == 0 or terminal is True) \
                and len(buffer) >= params['minibatch_size']:

            minibatch = buffer.random_sample(params['minibatch_size'])
            t = ProcessMinibatch(minibatch)

            with torch.no_grad():
                target_action, target_action_log = actor.action_selection(t.next_states)
                target_Q = torch.min(critic1.target_net(t.next_states, target_action),
                                     critic2.target_net(t.next_states, target_action)) - alpha * target_action_log
            target = t.rewards + gamma * (1 - t.terminals) * target_Q
            current_v1 = critic1.net(t.states, t.actions)
            current_v2 = critic2.net(t.states, t.actions)
            critic_loss1 = critic_loss_fnc(target, current_v1)
            critic_loss2 = critic_loss_fnc(target, current_v2)
            wandb.log({"value_loss": (critic_loss1+critic_loss2)/2, 'step': global_step, 'episode': episode},
                      commit=False)
            critic1.optimise(critic_loss1)
            critic2.optimise(critic_loss2)
            critic1.soft_target_update()
            critic2.soft_target_update()

            policy_action, policy_action_log = actor.action_selection(t.states)
            with torch.no_grad():
                Q_min = torch.min(critic1.net(t.states, policy_action), critic2.net(t.states, policy_action))
            actor_loss = (Q_min - alpha * policy_action_log).mean()
            wandb.log({"policy_loss": actor_loss,  'step': global_step, 'episode': episode}, commit=False)
            actor.optimise(-actor_loss)

    wandb.log({"episode_reward": episode_reward, "episode": episode})
