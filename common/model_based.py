import torch
import wandb
import matplotlib.pyplot as plt

from RL_framework.common.buffer import ProcessMinibatch


class DynamicsModel:
    def __init__(self, model, buffer, loss_func, opt, env, gamma):
        self.model = model
        self.buffer = buffer
        self.loss_func = loss_func
        self.opt = opt
        self.env = env
        self.gamma = gamma

    def populate_buffer_randomly(self, fraction):
        while len(self.buffer) < int(fraction * self.buffer.capacity):
            state = self.env.env.reset() + torch.normal(0, 0.01, size=(self.env.obs_size,)).numpy()
            terminal = False
            while terminal is False:
                if str(self.env.env.action_space)[:8] == 'Discrete':
                    action = torch.randint(self.env.action_size, size=(1,)).item()
                elif str(self.env.env.action_space)[:3] == 'Box':
                    action = self.env.action_low + (self.env.action_high - self.env.action_low) *\
                             torch.rand(self.env.action_size).numpy()
                next_state, reward, terminal, _ = self.env.env.step(action)
                self.buffer.add(state, action, reward, next_state, terminal, None, None)
                state = next_state

    def train_model(self, epochs, minibatch_size, model_type='forward'):
        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)
            t.standardise(self.env.obs_high)
            if model_type == 'forward':
                target_state = t.next_states + torch.normal(0, 0.001, size=t.states.shape)
            elif model_type == 'diff':
                target_state = t.next_states - t.states + torch.normal(0, 0.001, size=t.states.shape)
            target = torch.cat((target_state, t.terminals), dim=1)
            next_states, terminal = self.model(t.states + torch.normal(0, 0.001, size=t.states.shape),
                                 t.actions + torch.normal(0, 0.001, size=t.actions.shape))
            terminals = torch.argmax(terminal, dim=1).reshape(-1, 1)
            loss = self.loss_func(torch.cat((next_states, terminals), dim=1), target)
            wandb.log({"model_loss": loss}, commit=False)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

'''
    def plot_vector_field(self, dims, policy, n_pts=50):
        states = [torch.linspace(self.env.obs_low[i], self.env.obs_high[i], n_pts) for i in range(self.env.obs_size)]
        states_mesh = torch.meshgrid(states)
        states_input = torch.stack([states_mesh[i].reshape(-1, 1) for i in range(len(states_mesh))]).squeeze().T
        actions = policy(states_input)
        next_states = self.model(torch.cat((states_input, actions), dim=1))

        X = states_mesh[dims[0]][dims[0]]
        Y = states_mesh[dims[1]][dims[1]]
        x = states_input[:,0].reshape(states_mesh[0].shape)[:,1,:]
        y = states_input[:,dims[1]]
        new_x = next_states[:,dims[0]]
        new_y = next_states[:,dims[1]]
        dx = new_x - x
        dy = new_y - y
        # plot vector field and its intensity
        fig = plt.figure(figsize=(4, 4));
        ax = fig.add_subplot(111)
        ax.streamplot(X.numpy().T, Y.numpy().T, dx.detach().numpy().T, dy.detach().numpy().T, color='black')
        ax.contourf(X.T, Y.T, torch.sqrt(dx.detach().T ** 2 + dy.detach().T ** 2), cmap='RdYlBu')
        fig.show()
'''

class MPC:
    def __init__(self, model, env, gamma):
        self.model = model
        self.env = env
        self.gamma = gamma

    def random_shooting(self, state, k, horizon, normalise_state=True):
        samples = torch.randint(self.env.action_size, size=(k, horizon))
        rewards = torch.zeros(k)
        for i, sample in enumerate(samples):
            current_state = torch.Tensor(state)
            if normalise_state:
                current_state /= self.env.obs_high
            reward = 0
            for j, action in enumerate(sample):
                state_action = torch.cat((current_state, action.reshape(1)))
                with torch.no_grad():
                    current_state += self.model(state_action)
                if normalise_state:
                    reward += self.gamma ** j * self.env.reward_func(current_state * self.env.obs_high)
                else:
                    reward += self.gamma ** j * self.env.reward_func(current_state)
            rewards[i] = reward
        best_k = torch.argmax(rewards)
        return samples[best_k, 0].item()
