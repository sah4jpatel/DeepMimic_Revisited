import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os
import math


class Agent(nn.Module):
    def __init__(self, n_obs, n_actions, mean_state, std_state, load_models=False):
        super().__init__()
        self.name = 'PPO'
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.mean_state = mean_state
        self.std_state = std_state

        self.policy = nn.Sequential(
            nn.Linear(n_obs, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(n_obs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        if load_models:
            self.load_models()

    def save_models(self):
        torch.save(self.policy.state_dict(), 'models/policy_net4')
        torch.save(self.value.state_dict(), 'models/value_net4')

    def load_models(self):
        if os.path.exists('models/policy_net2') and os.path.exists('models/value_net4'):
            self.policy.load_state_dict(torch.load('models/policy_net4', weights_only=True))
            self.value.load_state_dict(torch.load('models/value_net4', weights_only=True))

    def get_loss(self, traj_data, first_epoch, epsilon=.9):
        # print(traj_data.states)
        # print("traj data states: " + str(traj_data.states))
        # print("mean state: " + str(self.mean_state))
        # print("std state: " + str(self.std_state))
        # print("input: " + str(((traj_data.states - self.mean_state) * self.std_state).shape))
        # logits = self.policy((traj_data.states.squeeze() - self.mean_state) * self.std_state)
        # print("inputs: " + str(traj_data.states.squeeze()[:,0]))
        logits = self.policy(traj_data.states.squeeze())
        # print("logits: " + str(logits))
        means = logits.squeeze()
        # print("means: " + str(means.shape))
        # print(means)
        cov = torch.diag_embed(torch.ones(means.shape) * 0.01)

        # print("cov: " + str(cov))
        # print("means: " + str(means))
        # print("cov: " + str(cov.shape))
        new_probs = MultivariateNormal(means, cov)
        # print("probs: " + str(new_probs))
        new_log_probs = new_probs.log_prob(traj_data.actions)

        A_t = traj_data.returns - self.value(traj_data.states).squeeze()
        value_loss = torch.mean(torch.square(A_t))

        old_log_probs = traj_data.log_probs
        # print("new log probs: " + str(new_log_probs))
        # print("old log probs: " + str(old_log_probs))
        policy_ratio = torch.exp(new_log_probs - old_log_probs)
        # print("policy ratio: " + str(policy_ratio))
        # policy_loss = -torch.mean(torch.mean(torch.minimum(
        #   policy_ratio * A_t,
        #   torch.clip(policy_ratio, 1.0-epsilon, 1.0+epsilon) * A_t
        # )))
        policy_loss = -torch.mean(torch.minimum(
          policy_ratio * A_t,
          torch.clip(policy_ratio, 1.0-epsilon, 1.0+epsilon) * A_t
        ))

        regularization_loss = sum([p.pow(2).sum() for p in self.policy.parameters()])
        loss = value_loss + policy_loss + 0.5 * regularization_loss
        if loss == float('inf'):
            print("policy loss: " + str(policy_loss))
            print("value loss: " + str(value_loss))
            print("policy ratio: " + str(policy_ratio))
            print("new log probs: " + str(new_log_probs))
            print("old log probs: " + str(old_log_probs))
            print("At: " + str(A_t))
            print("returns: " + str(traj_data.returns))
            print("value: " + str(self.value(traj_data.states).squeeze()))

        return loss

    def get_action(self, obs):
        # logits = self.policy((obs - self.mean_state) * self.std_state)
        # print("inputs2: " + str(obs[0]))
        logits = self.policy(obs)
        # print("logits2: " + str(logits))
        means = logits
        # print("means: " + str(means))
        # cov = torch.diag(torch.exp(logits[self.n_actions:]))
        cov = torch.diag(torch.ones(self.n_actions) * 0.01)
        probs = MultivariateNormal(means, cov)
        # print("probs 2: " + str(probs.))
        actions = torch.clamp(probs.sample(), -1.0, 1.0)
        # print("is nan: " + str(actions.isnan().any()))
        # print(torch.max(actions))
        # print(torch.min(actions))
        # print(torch.mean(actions))
        # print("actions: " + str(actions))
        return actions, probs