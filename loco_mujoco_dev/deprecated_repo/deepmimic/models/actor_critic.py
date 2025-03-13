# deepmimic/models/actor_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """Actor-Critic network architecture for PPO."""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        """
        Initialize actor-critic network.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(ActorCritic, self).__init__()
        
        # Actor network layers
        actor_layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Tanh()
            ])
            prev_dim = dim
            
        self.actor_body = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network layers
        critic_layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Tanh()
            ])
            prev_dim = dim
            
        self.critic_body = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(prev_dim, 1)
        
    def forward(self, obs):
        """Forward pass through both actor and critic networks."""
        # Actor forward pass
        actor_features = self.actor_body(obs)
        action_mean = self.actor_mean(actor_features)
        action_std = torch.exp(self.actor_log_std)
        
        # Create normal distribution
        action_dist = Normal(action_mean, action_std)
        
        # Critic forward pass
        critic_features = self.critic_body(obs)
        value = self.critic_head(critic_features)
        
        return action_dist, value
        
    def get_action(self, obs, deterministic=False):
        """Sample action from policy distribution."""
        action_dist, value = self(obs)
        
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
            
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
        
    def evaluate_actions(self, obs, actions):
        """Evaluate actions and compute log probs, values, and entropy."""
        action_dist, values = self(obs)
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().mean()
        
        return log_probs, values, entropy
