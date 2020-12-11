import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import random


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

    
class PolicyFunction(nn.Module):
    def __init__(self, net, opt, learning_rate):
        super(PolicyFunction, self).__init__() 
        self.net = net
        self.optimiser = opt(net.parameters(), lr=learning_rate)
        
    def softmax_action(self, state):
        probs = self.net(torch.from_numpy(state).float())
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action, probs[action].log().reshape(1)
    
    def get_policy(self, state):
        return self.net(torch.from_numpy(state).float())
    
    def log_prob(self, state, action):
        policy = self.net(torch.from_numpy(state).float())
        return torch.log(policy[action]) 
    
    def optimise(self, loss, grad_clamp=False):
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)

        if grad_clamp is True:  # check if works
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimiser.step()
  
   
class ValueFunction(nn.Module):
    def __init__(self, net, opt, learning_rate, loss_function, epsilon=None, target_net=False):
        super(ValueFunction, self).__init__() 
        self.net = net
        self.optimiser = opt(net.parameters(), lr=learning_rate)
        self.loss_function = loss_function
        self.epsilon = epsilon
        if target_net is True:
            self.target_net = net
    
    def optimise(self, target, current_v, grad_clamp=False):
        loss = self.loss_function(target, current_v)
        wandb.log({"value_loss": loss}, commit=False)
        
        self.optimiser.zero_grad()
        loss.backward()
        
        if grad_clamp is True:  # check if works
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
                
        self.optimiser.step()
        return 
        
    def epsilon_greedy_action(self, state, episode):
        
        epsilon = self.epsilon['eps_end'] + (self.epsilon['eps_start'] - self.epsilon['eps_end']) \
                  * np.exp(-episode / self.epsilon['eps_decay'])
        wandb.log({"epsilon": epsilon}, commit=False)
        
        if np.random.rand() < epsilon:
            return random.randrange(self.net.layers[-1].out_features)
        else:
            with torch.no_grad():
                return torch.argmax(self.net(torch.from_numpy(state).float())).item()
            
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
