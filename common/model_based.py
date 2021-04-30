import torch
import wandb
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

    def train_model(self, epochs, minibatch_size, model_type='forward', standardise=False, noise=True,
                    log_commit=False):

        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)

            if standardise:
                t.standardise(self.env.obs_high)

            if model_type == 'forward':
                target = t.next_states
            elif model_type == 'diff':
                target = t.next_states - t.states

            if noise is True:
                target += torch.normal(0, 0.01, size=t.states.shape)
                t.states += torch.normal(0, 0.01, size=t.states.shape)
                t.actions += torch.normal(0, 0.01, size=t.actions.shape)

            next_states = self.model(torch.cat((t.states, t.actions), dim=1))

            loss = self.loss_func(next_states, target)
            wandb.log({"model_loss": loss}, commit=log_commit)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


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
