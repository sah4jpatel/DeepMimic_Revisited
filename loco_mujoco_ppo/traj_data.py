import torch
from copy import deepcopy

class TrajData:
    def __init__(self, n_steps, n_envs, n_obs, n_actions):
        s, e, o, a = n_steps, n_envs, n_obs, n_actions
        from torch import zeros

        self.states = zeros((s, e, o))
        self.actions = zeros((s, e, a))
        self.rewards = zeros((s, e))
        self.not_dones = zeros((s, e))

        self.log_probs = zeros((s, e))
        self.returns = zeros((s, e))

        self.n_steps = s

    def detach(self):
        self.actions = self.actions.detach()
        self.log_probs = self.log_probs.detach()

    def store(self, t, s, a, r, lp, d):
        self.states[t] = s
        self.actions[t] = a
        self.rewards[t] = torch.Tensor(r)

        self.log_probs[t] = lp
        self.not_dones[t] = 1 - torch.Tensor(d)

    def calc_returns(self, gamma = .99):
        self.returns = deepcopy(self.rewards)

        for t in reversed(range(self.n_steps-1)):
            self.returns[t] += self.returns[t+1] * self.not_dones[t] * gamma