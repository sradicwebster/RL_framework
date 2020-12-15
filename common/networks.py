import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import random
import copy


class SequentialNetwork(nn.Module):
    def __init__(self, layers):
        super(SequentialNetwork, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)


class DuelingNetwork(nn.Module):
    def __init__(self, nodes):  # nodes = [obs_size, 64, 128, n_actions]
        super(DuelingNetwork, self).__init__()

        self.fc1 = nn.Linear(nodes[0], nodes[1])
        self.fc_value = nn.Linear(nodes[1], nodes[2])
        self.fc_adv = nn.Linear(nodes[1], nodes[2])

        self.value = nn.Linear(nodes[2], 1)
        self.adv = nn.Linear(nodes[2], nodes[-1])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=0, keepdim=True)
        Q = value + adv - advAverage

        return Q


class Qnet_continuous_actions(nn.Module):
    #  add layer norm??
    def __init__(self, obs_size, action_n):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32 + action_n, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CommonFunctions:
    def __init__(self, net, optimiser, target_net, tau):
        self.net = net
        self.optimiser = optimiser
        if target_net is True:
            self.target_net = copy.deepcopy(net)
        self.tau = tau

    def optimise(self, loss, grad_clamp=False):
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)

        if grad_clamp is True:
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimiser.step()

    def hard_target_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)

    def soft_target_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)


class PolicyFunction(CommonFunctions):
    def __init__(self, net, optimiser, target_net=False, tau=None):
        super().__init__(net, optimiser, target_net, tau)

    def softmax_action(self, state):
        probs = self.net(torch.from_numpy(state).float().reshape(1, -1))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action, probs.squeeze()[action].log()

    def get_policy(self, state):
        return self.net(torch.from_numpy(state).float())

    def log_prob(self, state, action):
        policy = self.net(torch.from_numpy(state).float())
        return torch.log(policy[action])


class ValueFunction(CommonFunctions):
    def __init__(self, net, optimiser, target_net=False, tau=None, epsilon=None):
        super().__init__(net, optimiser, target_net, tau)
        self.epsilon = epsilon

    def epsilon_greedy_action(self, state, episode):

        epsilon = self.epsilon['eps_end'] + (self.epsilon['eps_start'] - self.epsilon['eps_end']) \
                  * np.exp(-episode / self.epsilon['eps_decay'])
        wandb.log({"epsilon": epsilon}, commit=False)

        if np.random.rand() < epsilon:
            return random.randrange(self.net.layers[-1].out_features)
        else:
            with torch.no_grad():
                return torch.argmax(self.net(torch.from_numpy(state).float())).item()