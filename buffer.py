from collections import namedtuple
import random
import torch


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'step',
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


class ProcessMinibatch:
    def __init__(self, minibatch, gamma):
        self.gamma = gamma
        self.states, self.actions, self.rewards, self.next_states, self.terminals, self.steps, self.action_log_prob \
            = [], [], [], [], [], [], []
        for transition in minibatch:
            self.states.append(transition.state)
            self.actions.append(transition.action)
            self.rewards.append(transition.reward)
            self.next_states.append(transition.next_state)
            self.terminals.append(transition.terminal)
            self.steps.append(transition.step)
            self.action_log_prob.append(transition.action_log_prob)

        self.rewards = torch.Tensor(self.rewards).reshape(-1, 1)
        if self.action_log_prob[0] is not None:
            self.action_log_prob = torch.stack(self.action_log_prob)

    def one_step_target(self, critic):
        target = [self.rewards[i] + (1 - self.terminals[i]) * self.gamma *
                  critic(torch.from_numpy(self.next_states[i]).float())
                  for i in range(len(self.rewards))]
        return torch.stack(target)

    def qlearning_target(self, critic):
        target = [self.rewards[i] + (1 - self.terminals[i]) * self.gamma * critic(
            torch.from_numpy(self.next_states[i]).float()).max() for i in range(len(self.rewards))]
        return torch.stack(target)

    def current_value(self, critic):
        if critic.layers[-1].out_features == 1:
            current_v = [critic(torch.from_numpy(self.states[i]).float()) for i in range(len(self.states))]
        else:
            current_v = [critic(torch.from_numpy(self.states[i]).float())[self.actions[i]] for i in
                         range(len(self.states))]
        return torch.stack(current_v).reshape(-1,1)

    def td_error(self, critic):
        target = self.one_step_target(critic)
        current_v = self.current_value(critic)
        return target - current_v

    def discount_gamma(self):
        return torch.Tensor([self.gamma ** self.steps[i] for i in range(len(self.steps))]).reshape(-1, 1)

    def log_prob(self, actor, i=-1):
        if i == -1:
            log_probs = [actor(torch.from_numpy(self.states[i]).float())[self.actions[i]].log().reshape(1)
                         for i in range(len(self.states))]
            return torch.stack(log_probs)
        else:
            return actor(torch.from_numpy(self.states[i]).float())[self.actions[i]].log().reshape(1)

    def reward_to_go(self):
        discounted_gamma = self.discount_gamma()
        discounted_rewards = torch.Tensor(self.rewards).reshape(-1, 1) * discounted_gamma

        return torch.stack([sum(discounted_rewards[i:]) for i in range(len(discounted_rewards))])

    def discounted_cumsum(self, input):
        discounted_gamma = self.discount_gamma()
        discounted_input = input * discounted_gamma
        return torch.stack([sum(discounted_input[i:]) for i in range(len(discounted_input))])
