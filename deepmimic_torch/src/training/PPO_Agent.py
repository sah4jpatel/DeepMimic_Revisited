
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('/mnt/c/Users/cnikh/Projects/deepmimic_torch/src/training')
from ActorCritic import ActorCritic

#%%
class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, train_iters=10):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        
        self.actor_critic = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * (1 - dones[step]) * next_value - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def update(self, obs, actions, log_probs_old, returns, advantages):
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(self.train_iters):
            
            new_log_probs, entropy, value = self.actor_critic.evaluate_actions(obs, actions)
            ratio = torch.exp(new_log_probs - log_probs_old)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = (returns - value).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(obs)
        return action.numpy(), log_prob.item(), value.item()

    def save(self, filepath):
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully.")

# %%
