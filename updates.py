import torch


class ProcessMinibatch:
    def __init__(self, minibatch):
        self.states, self.actions, self.rewards, self.next_states, self.terminals, self.steps = [], [], [], [], [], []
        for transition in minibatch:
            self.states.append(transition.state)
            self.actions.append(transition.action)
            self.rewards.append(transition.reward)
            self.next_states.append(transition.next_state)
            self.terminals.append(transition.terminal)
            self.steps.append(transition.step)

    def one_step_target(self, net, gamma):
        target = [self.rewards[i] + (1 - self.terminals[i]) * gamma * net(torch.from_numpy(self.next_states[i]).float())
                  for i in range(len(self.rewards))]
        return torch.stack(target)

    def qlearning_target(self, qnet, gamma):  # check if works
        target = [self.rewards[i] + (1 - self.terminals[i]) * gamma * qnet(
            torch.from_numpy(self.next_states[i]).float()).max() for i in range(len(self.rewards))]
        return torch.stack(target)

    def current_value(self, net):
        if net.layers[-1].out_features == 1:
            current_v = [net(torch.from_numpy(self.states[i]).float()) for i in range(len(self.states))]
        else:
            current_v = [net(torch.from_numpy(self.states[i]).float())[self.actions[i]] for i in
                         range(len(self.states))]
        return torch.stack(current_v)

    def discount(self, gamma):
        return [gamma ** self.steps[i] for i in range(len(self.steps))]

    def log_prob(self, actor):
        log_probs = []
        for i in range(len(self.states)):
            policy = actor.get_policy(self.states[i])
            log_probs.append(torch.log(policy[self.actions[i]]))
        return torch.stack(log_probs)
