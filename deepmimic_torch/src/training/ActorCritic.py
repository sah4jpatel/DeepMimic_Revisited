#%%

import torch
import torch.nn as nn
from torch.distributions import Normal

import sys
sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/src/arch')
from mlp import MLP
from attention import AttentionMLP

#%%

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super(ActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy = MLP(self.obs_dim, hidden_size, self.act_dim, tanh=True)
        self.value = MLP(self.obs_dim, hidden_size, 1)
        # self.policy = AttentionMLP(self.obs_dim, hidden_size, self.act_dim, num_heads=4)
        # self.value = AttentionMLP(self.obs_dim, hidden_size, 1, num_heads=4)

        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, x):
        mean = self.policy(x)
        value = self.value(x)
        return mean, value
    
    def get_action(self, obs):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, entropy, value




#%%

if __name__ == "__main__":
    ac = ActorCritic(7, 28)
    obs = torch.randn(16, 7)
    action, log_prob, value = ac.get_action(obs)
    print(action)
    print(log_prob)
    print(value)

    agent = PPOAgent(obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, train_iters=10)


    

# %%
