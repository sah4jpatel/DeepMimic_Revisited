import torch
from copy import deepcopy

class TrajData:
    def __init__(self, n_steps, n_envs, n_obs, n_actions,idx,ref_start_ind):
        s, e, o, a = n_steps, n_envs, n_obs, n_actions
        from torch import zeros

        self.traj_ind = idx

        self.states = zeros((s, e, o))
        self.actions = zeros((s, e, a))
        self.rewards = zeros((s, e))
        self.not_dones = zeros((s, e),dtype=torch.int32)

        self.log_probs = zeros((s, e))
        self.returns = zeros((s, e))
        # self.advantages = zeros((s,e))

        self.n_steps = s
        self.ref_motion_start_ind = ref_start_ind

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

        # # normalize returns
        # self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)
        print(torch.max(self.returns),torch.min(self.returns))

