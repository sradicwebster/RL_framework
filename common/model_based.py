import torch
import wandb
from RL_framework.common.buffer import ProcessMinibatch


class DynamicsModel:
    def __init__(self, model, buffer, loss_func, opt, env, model_type='diff'):
        self.model = model
        self.buffer = buffer
        self.loss_func = loss_func
        self.opt = opt
        self.env = env
        self.type = model_type

    def train_model(self, epochs, minibatch_size, standardise=False, noise_std=None):

        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)

            if standardise:
                t.standardise(self.env.obs_high)

            if self.type == 'forward':
                target = t.next_states
            elif self.type == 'diff':
                target = t.next_states - t.states

            if noise_std is not None:
                target += torch.normal(0, noise_std, size=t.states.shape)
                t.states += torch.normal(0, noise_std, size=t.states.shape)
                t.actions += torch.normal(0, noise_std, size=t.actions.shape)

            current = self.model(torch.cat((t.states, t.actions), dim=1))

            loss = self.loss_func(current, target)
            wandb.log({"model_loss": loss}, commit=False)
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
