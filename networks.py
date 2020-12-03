import torch
import torch.nn as nn
import numpy as np
import wandb
import random


class SequentialNetwork(nn.Module):
    def __init__(self, layers):
        super(SequentialNetwork, self).__init__() 
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): return self.layers(x)
    
    
class PolicyFunction(nn.Module):
    def __init__(self, net, opt, learning_rate):
        super(PolicyFunction, self).__init__() 
        self.net = net
        self.optimiser = opt(net.parameters(), lr=learning_rate)
        
    def softmax_action(self, state):
        probs = self.net(torch.from_numpy(state).float())
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action
    
    def get_policy(self, state):
        return self.net(torch.from_numpy(state).float())
    
    def log_prob(self, state, action):
        policy = self.net(torch.from_numpy(state).float())
        return torch.log(policy[action]) 
    
    def optimise(self, loss):
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)
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
