import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os


class Agent(nn.Module):
    def __init__(self, n_obs, n_actions, load_models=False):
        super().__init__()
        self.name = 'PPO'
        self.n_obs = n_obs
        self.n_actions = n_actions

        self.policy = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)
        )
        self.old_policy_parameters = None
        if load_models:
            self.load_models()

    def save_models(self):
        torch.save(self.policy.state_dict(), './models/policy_net')
        torch.save(self.value.state_dict(), './models/value_net')

    def load_models(self):
        if os.path.exists('models/policy_net') and os.path.exists('models/value_net'):
            self.policy.load_state_dict(torch.load('models/policy_net', weights_only=True))
            self.value.load_state_dict(torch.load('models/value_net', weights_only=True))

    def get_loss(self, traj_data, epsilon=.1):
        # print(traj_data.states)
        logits = self.policy(traj_data.states)
        # print(logits)
        # print("logits: " + str(logits.shape))
        means = logits.squeeze()
        # print(means)
        cov = torch.diag_embed(torch.ones(logits.shape) * 0.2)
        # print("means: " + str(means))
        # print("cov: " + str(cov.shape))
        new_probs = MultivariateNormal(means, cov)
        new_log_probs = new_probs.log_prob(traj_data.actions)

        A_t = traj_data.returns - self.value(traj_data.states).squeeze()
        value_loss = torch.mean(torch.mean(torch.square(A_t)))

        policy_ratio = torch.exp(new_log_probs - traj_data.log_probs)
        policy_loss = -torch.mean(torch.mean(torch.minimum(
          policy_ratio * A_t,
          torch.clip(policy_ratio, 1.0-epsilon, 1.0+epsilon) * A_t
        )))
        loss = value_loss + policy_loss

        self.old_policy_parameters = self.policy.state_dict()

        return loss

    def get_action(self, obs):
        logits = self.policy(obs)
        means = logits
        # cov = torch.diag(torch.exp(logits[self.n_actions:]))
        cov = torch.diag(torch.ones(self.n_actions) * 0.2)
        probs = MultivariateNormal(means, cov)
        actions = torch.clamp(probs.sample(), -1.0, 1.0)
        # print("is nan: " + str(actions.isnan().any()))
        # print(torch.max(actions))
        # print(torch.min(actions))
        # print(torch.mean(actions))
        # print("actions: " + str(actions))
        return actions, probs