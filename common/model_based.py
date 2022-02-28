import torch
from torch.distributions.normal import Normal
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import wandb
from RL_framework.common.buffer import ProcessMinibatch


class DynamicsModel:
    def __init__(self, model, buffer, loss_func, opt, env, model_type='diff', reward=None, rew_opt=None):
        self.model = model
        self.buffer = buffer
        self.loss_func = loss_func
        self.opt = opt
        self.env = env
        self.type = model_type
        self.reward = reward
        self.rew_opt = rew_opt

    def train_model(self, epochs, minibatch_size, grad_steps=1, standardise=False, noise_std=None):

        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)

            if standardise:
                t.standardise(self.env.obs_high)

            if self.type == 'forward':
                target = t.next_states
            else:
                target = t.next_states - t.states

            if noise_std is not None:
                target += torch.normal(0, noise_std, size=t.states.shape)
                t.states += torch.normal(0, noise_std, size=t.states.shape)
                t.actions += torch.normal(0, noise_std, size=t.actions.shape)

            for _ in range(grad_steps):
                current = self.model(torch.cat((t.states, t.actions), dim=1))
                loss = self.loss_func(current, target)
                wandb.log({"model_loss": loss}, commit=False)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def train_reward_fnc(self, epochs, minibatch_size):

        for i in range(epochs):
            minibatch = self.buffer.random_sample(minibatch_size)
            t = ProcessMinibatch(minibatch)
            target = t.rewards
            current = self.reward(torch.cat((t.states, t.actions), dim=1))
            loss = self.loss_func(current, target)
            wandb.log({"reward_loss": loss}, commit=False)
            self.rew_opt.zero_grad()
            loss.backward()
            self.rew_opt.step()


class MPC:
    def __init__(self, model_net, env, gamma, reward_net=None):
        self.model = model_net
        self.env = env
        self.gamma = gamma
        self.reward_net = reward_net
        self.action_sequence = None

    def _rollout(self, state, samples, normalise, grad=False):
        rewards = torch.zeros(len(samples))
        state = torch.ones(len(samples), len(state)) * torch.Tensor(state)

        if normalise:
            state /= self.env.obs_high

        for t in range(samples.shape[1]):
            action = samples[:, t]
            if grad is False:
                with torch.no_grad():
                    state += self.model(torch.cat((state, action), dim=1))
                rewards += torch.Tensor([self.gamma ** t * self.env.reward_func(state[i], action[i])
                                         for i in range(len(state))])
            else:
                rewards.requires_grad_()
                state += self.model(torch.cat((state, action), dim=1))
                rewards = rewards.add(self.gamma ** t * self.reward_net(torch.cat((state, action), dim=1)).squeeze())

        return rewards

    def random_shooting(self, state, k, horizon, normalise_state=False):
        samples = torch.randint(self.env.action_size, size=(k, horizon))
        rewards = self._rollout(state, samples, normalise_state)
        best_k = torch.argmax(rewards)
        return samples[best_k, 0].item()

    def cem_planning(self, state, k, horizon, best_k, episode_step, keep_best_k=False, cem_iters=10, alpha=0.6,
                     normalise_state=False, grad=False, grad_iters=5, grad_clip=True):

        if episode_step == 0:
            mu = torch.zeros(horizon, self.env.action_size)
        else:
            mu = torch.cat((self.action_sequence[1:], torch.Tensor([[0]])))
        sigma = torch.ones_like(mu) / 2

        for i in range(cem_iters):
            action_dis = Normal(mu, sigma)
            if keep_best_k is True and i != 0:
                samples = torch.cat((best_samples, torch.clamp(action_dis.sample(((k - best_k),)), -1, 1)))
            else:
                samples = torch.clamp(action_dis.sample((k,)), -1, 1)

            if grad is True:
                samples = samples.requires_grad_()
                action_opt = optim.Adam([samples], lr=0.05)
                for _ in range(grad_iters):
                    action_opt.zero_grad()
                    rewards = self._rollout(state, samples, normalise_state, grad=grad).sum()
                    (-rewards).backward()
                    if grad_clip is True:
                        clip_grad_norm_(samples.grad.data, max_norm=1, norm_type=2)
                    action_opt.step()
                samples = samples.detach()

            rewards = self._rollout(state, samples, normalise_state, grad=False)
            best_samples = samples[rewards.topk(best_k, sorted=False).indices]
            mu_elite, sigma_elite = torch.std_mean(best_samples, dim=0, unbiased=False)
            mu = alpha * mu + (1 - alpha) * mu_elite
            sigma = alpha * sigma + (1 - alpha) * sigma_elite
        self.action_sequence = mu
        return mu[0]
