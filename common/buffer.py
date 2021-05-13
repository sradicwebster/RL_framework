from collections import namedtuple
import random
import torch
import wandb


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'episode_step',
                                       'action_log_prob'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self): return len(self.memory)

    def add(self, *args):
        """Save a transition."""

        # Extend memory if capacity not yet reached.
        if self.__len__() < self.capacity:
            self.memory.append(None)

        # Overwrite current entry at this position.
        self.memory[self.position] = Transition(*args)

        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def random_sample(self, batch_size):
        """Retrieve a random sample of transitions."""
        assert batch_size <= self.__len__()

        return random.sample(self.memory, batch_size)

    def ordered_sample(self, batch_size):
        assert batch_size <= self.__len__()

        return self.memory[:batch_size]

    def empty(self):
        self.memory = []
        self.position = 0

    def populate_randomly(self, env, fraction, step=0):
        while self.__len__() < int(fraction * self.capacity):
            state = env.env.reset()
            terminal = False
            while terminal is False:
                if str(env.env.action_space)[:8] == 'Discrete':
                    action = torch.randint(env.action_size, size=(1,)).numpy()
                    next_state, reward, terminal, _ = env.env.step(action)
                elif str(env.env.action_space)[:3] == 'Box':
                    action = torch.rand(env.action_size)
                    action_scaled = env.action_low + (env.action_high - env.action_low) * action
                    next_state, reward, terminal, _ = env.env.step(action_scaled)
                    action = (action - 0.5) * 2  # action in [-1, 1]
                else:
                    print('Action space not implemented')

                step += 1
                self.add(state, action, reward, next_state, terminal, None, None)
                state = next_state
        return step


class ProcessMinibatch:
    def __init__(self, minibatch):
        self.states, self.actions, self.rewards, self.next_states, self.terminals, self.steps, self.action_log_prob \
            = [], [], [], [], [], [], []
        for transition in minibatch:
            self.states.append(transition.state)
            self.actions.append(transition.action)
            self.rewards.append(transition.reward)
            self.next_states.append(transition.next_state)
            self.terminals.append(transition.terminal)
            self.steps.append(transition.episode_step)
            self.action_log_prob.append(transition.action_log_prob)

        self.states = torch.Tensor(self.states)
        if type(self.actions[0]) == int or self.actions[0].shape[0] == 1:
            self.actions = torch.tensor(self.actions).reshape(-1, 1)
        else:
            self.actions = torch.tensor(self.actions)
        self.rewards = torch.Tensor(self.rewards).reshape(-1, 1)
        self.next_states = torch.Tensor(self.next_states)
        self.terminals = torch.Tensor(self.terminals).reshape(-1, 1)
        if self.steps[0] is not None:
            self.steps = torch.Tensor(self.steps).reshape(-1, 1)
        if self.action_log_prob[0] is not None:
            self.action_log_prob = torch.stack(self.action_log_prob)

    def standardise(self, obs_max):
        self.states /= obs_max
        self.next_states /= obs_max

