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
        self.obs_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.gamma = gamma

    def populate_buffer(self, policy):
        while len(self.buffer) < self.buffer.capacity:
            state = self.env.reset() + torch.normal(0, 0.001, size=(self.obs_size,)).numpy()
            terminal = False
            while terminal is False:
                action = policy
                # action = torch.randint(self.action_size)
                next_state, reward, terminal, _ = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, terminal, None, None)
                state = next_state

    def train_model(self, epochs, minibatch_size, target='forward'):
        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)
            t.standardise(self.env.obs_max)
            if target == 'forward':
                target = t.next_states - t.states + torch.normal(0, 0.001, size=t.states.shape)
            elif target == 'diff':
                target = t.next_states + torch.normal(0, 0.001, size=t.states.shape)
            state_actions = torch.cat((t.states, t.actions), dim=1)
            current = self.model(state_actions + torch.normal(0, 0.001, size=state_actions.shape))
            loss = self.loss_func(target, current)
            wandb.log({"model_loss": loss})
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
                current_state /= self.env.obs_max
            reward = 0
            for j, action in enumerate(sample):
                state_action = torch.cat((current_state, action.reshape(1)))
                with torch.no_grad():
                    current_state += self.model(state_action)
                if normalise_state:
                    reward += self.gamma ** j * self.env.reward_func(current_state * self.env.obs_max)
                else:
                    reward += self.gamma ** j * self.env.reward_func(current_state)
            rewards[i] = reward
        best_k = torch.argmax(rewards)
        return samples[best_k, 0].item()
