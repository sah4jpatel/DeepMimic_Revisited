# deepmimic/ppo/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from deepmimic.models.actor_critic import ActorCritic

class PPO:
    """Proximal Policy Optimization with Generalized Advantage Estimation."""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256], 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_ratio=0.2, target_kl=0.01, value_coef=0.5, 
                 entropy_coef=0.01, max_grad_norm=0.5,
                 update_epochs=10, mini_batch_size=64,
                 device=None):
        """
        Initialize PPO algorithm.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            target_kl: Target KL divergence
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs per PPO update
            mini_batch_size: Mini-batch size for updates
            device: Device to run on (CPU or GPU)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Create actor-critic network
        self.ac = ActorCritic(obs_dim, action_dim, hidden_dims).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
    def update(self, buffer):
        """
        Update policy using PPO.
        
        Args:
            buffer: Buffer containing collected experiences
            
        Returns:
            Dictionary with training metrics
        """
        # Check and fix device mismatches
        self._check_and_fix_device_mismatch()
            
        # Convert buffer data to tensors
        obs = torch.FloatTensor(buffer['obs']).to(self.device)
        actions = torch.FloatTensor(buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer['log_probs']).to(self.device)
        returns = torch.FloatTensor(buffer['returns']).to(self.device)
        advantages = torch.FloatTensor(buffer['advantages']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        metrics = {
            'value_loss': 0,
            'policy_loss': 0,
            'entropy': 0,
            'kl': 0,
            'clip_fraction': 0
        }
        
        # PPO update loop
        for epoch in range(self.update_epochs):
            # Generate random indices
            indices = np.random.permutation(len(obs))
            
            # Process mini-batches
            for start in range(0, len(obs), self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                # Get mini-batch data
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Get new log probs, values, and entropy
                new_log_probs, values, entropy = self.ac.evaluate_actions(mb_obs, mb_actions)
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Compute surrogate objectives
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # Value loss - ensure shape compatibility
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns.squeeze(-1))
                
                # Policy loss with entropy bonus
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Compute KL divergence
                approx_kl = ((mb_old_log_probs - new_log_probs).exp() - 1 - (mb_old_log_probs - new_log_probs)).mean().item()
                
                # Compute clip fraction
                clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                
                # Update metrics
                metrics['value_loss'] += value_loss.item() / (self.update_epochs * ((len(obs) - 1) // self.mini_batch_size + 1))
                metrics['policy_loss'] += policy_loss.item() / (self.update_epochs * ((len(obs) - 1) // self.mini_batch_size + 1))
                metrics['entropy'] += entropy.item() / (self.update_epochs * ((len(obs) - 1) // self.mini_batch_size + 1))
                metrics['kl'] += approx_kl / (self.update_epochs * ((len(obs) - 1) // self.mini_batch_size + 1))
                metrics['clip_fraction'] += clip_fraction / (self.update_epochs * ((len(obs) - 1) // self.mini_batch_size + 1))
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Early stopping based on KL divergence
                if approx_kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to high KL: {approx_kl:.4f}")
                    break
            
            # Early stopping across all epochs if KL is too high
            if metrics['kl'] > 2.0 * self.target_kl:
                print(f"Stopping all PPO updates early due to very high KL: {metrics['kl']:.4f}")
                break
                    
        return metrics
        
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute returns and advantages using GAE.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for the state after the last one
            
        Returns:
            Tuple of arrays (returns, advantages)
        """
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        # Add next_value to values for convenience
        values_extended = np.append(values, next_value)
        
        # Compute GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_extended[t+1] * (1 - dones[t]) - values_extended[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
            advantages[t] = last_gae_lam
            
        # Compute returns
        returns = advantages + values
        
        return returns, advantages
        
    def collect_rollout(self, env, num_steps, render=False):
        """
        Collect a rollout from the environment.
        
        Args:
            env: Environment to collect from
            num_steps: Number of steps to collect
            render: Whether to render the environment
            
        Returns:
            Buffer containing collected experiences
        """
        # Check and fix device mismatches
        self._check_and_fix_device_mismatch()
        
        # Initialize buffer
        buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
        # Reset environment
        obs = env.reset()
        
        for _ in range(num_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action, log_prob, and value
            with torch.no_grad():
                action, log_prob, value = self.ac.get_action(obs_tensor)
                
            # Convert to numpy
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().squeeze()
            value_np = value.cpu().numpy().squeeze()
            
            # Step environment
            next_obs, reward, done, info = env.step(action_np)
            
            # Render if requested
            if render:
                env.render()
                
            # Store transition
            buffer['obs'].append(obs)
            buffer['actions'].append(action_np)
            buffer['rewards'].append(reward)
            buffer['dones'].append(done)
            buffer['values'].append(value_np)
            buffer['log_probs'].append(log_prob_np)
            
            # Update observation
            obs = next_obs
            
            # Reset if done
            if done:
                obs = env.reset()
                
        # Get final value for bootstrapping
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, final_value = self.ac.get_action(obs_tensor)
            final_value = final_value.cpu().numpy().squeeze()
            
        # Compute returns and advantages
        returns, advantages = self.compute_gae(
            np.array(buffer['rewards']),
            np.array(buffer['values']),
            np.array(buffer['dones']),
            final_value
        )
        
        # Add returns and advantages to buffer
        buffer['returns'] = returns
        buffer['advantages'] = advantages
        
        # Convert lists to arrays
        for key in buffer:
            buffer[key] = np.array(buffer[key])
            
        return buffer
        
    def save(self, path):
        """Save model to path."""
        try:
            torch.save({
                'model_state_dict': self.ac.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, path, weights_only=True)
        except TypeError:
            # For older PyTorch versions without weights_only
            torch.save({
                'model_state_dict': self.ac.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        
    def load(self, path):
        """Load model from path."""
        self._check_and_fix_device_mismatch()
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # For older PyTorch versions
            checkpoint = torch.load(path, map_location=self.device)
            
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Make sure optimizer states are on the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
                    
    def _check_and_fix_device_mismatch(self):
        """Check if all parameters are on the correct device and fix if needed."""
        device_issues = False
        for name, param in self.ac.named_parameters():
            if param.device != self.device:
                print(f"Parameter {name} on {param.device}, moving to {self.device}")
                device_issues = True
                
        if device_issues:
            # Move model to the correct device
            self.ac = self.ac.to(self.device)
            
            # Recreate optimizer to ensure its states are on the correct device
            old_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = optim.Adam(self.ac.parameters(), lr=old_lr)
