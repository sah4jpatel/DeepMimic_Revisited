import torch
import torch.optim as optim
import torch.distributions as distributions

class PPOAgent:
    def __init__(self, policy, value, lr=3e-4, gamma=0.95, lam=0.95, epsilon=0.2, batch_size = 256, batches = 1, ss_val=1e-2, ss_policy=5e-5, momentum=0.9, T = 20):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.device)
        self.value = value.to(self.device)
        # self.optim_policy = optim.Adam(policy.parameters(), lr=lr)
        # self.optim_value = optim.Adam(value.parameters(), lr=lr)
        self.optim_policy = optim.SGD(policy.parameters(), lr=lr, momentum=momentum)
        self.optim_value = optim.SGD(value.parameters(), lr=lr, momentum=momentum)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.entropy_weight = 0.01
        self.std = 0.1
        self.T = 20

        self.batch_size = batch_size
        self.batches = batches


    def get_action(self, state):
        # state = torch.tensor(state, dtype=torch.float32)
        action_mean = self.policy(state.to(torch.float32))
        action_dist = distributions.Normal(action_mean, self.std)  # Gaussian policy
        action = action_dist.sample()
        return action.detach(), action_dist.log_prob(action).sum(dim=-1)
    
    def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
        """
        Computes the Generalized Advantage Estimation (GAE).

        Args:
            rewards (np.array): Array of rewards from rollout.
            values (np.array): Array of state values from the value network.
            gamma (float): Discount factor.
            lambda_ (float): GAE smoothing parameter.

        Returns:
            advantages (np.array): Computed advantage values.
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # Iterate backward for advantage computation
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (values[t + 1] if t < len(rewards) - 1 else 0) - values[t]
            advantages[t] = last_advantage = delta + gamma * lambda_ * last_advantage

        return advantages

    def get_minibatches(self, states, actions, rewards, returns, log_probs, next_states, dones):
        rollout_size = states.shape[0]
        indices = torch.randperm(rollout_size)  # Shuffle all indices

        for i in range(self.batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield states[batch_indices], actions[batch_indices], rewards[batch_indices], returns[batch_indices], log_probs[batch_indices], next_states[batch_indices], dones[batch_indices]

    def compute_td_lambda_targets(rewards, values, dones, gamma=0.99, lam=0.95):
        """
        Compute TD(lambda) target values for value function update.
        
        Args:
            rewards (Tensor): Shape (T,), rewards at each step.
            values (Tensor): Shape (T+1,), estimated state values from V_phi (bootstrapped).
            dones (Tensor): Shape (T,), 1 if episode ended, else 0.
            gamma (float): Discount factor.
            lam (float): Lambda factor for TD(lambda).
        
        Returns:
            Tensor: TD(lambda) target values (T,).
        """
        T = rewards.shape[0]
        td_lambda_targets = torch.zeros_like(rewards)
        G = values[-1]  # Bootstrap from last value
        
        for t in reversed(range(T)):  # Compute from end to start
            G = rewards[t] + gamma * (1 - dones[t]) * ((1 - lam) * values[t + 1] + lam * G)
            td_lambda_targets[t] = G
        
        return td_lambda_targets

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def log_prob(self, state, action):
        mean = self.policy(state)  # Output is the mean action
        std = torch.ones_like(mean) * self.std  # Example: fixed std (could also be learned)

        dist = torch.distributions.Normal(mean, std)  # Create Gaussian distribution
        log_prob = dist.log_prob(action).sum(dim=-1)  # Compute log probability

        return log_prob

    def update(self, states, actions, rewards, returns, log_probs, next_states, dones):
        states = torch.stack(states).to(torch.float32)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        o_log_probs = torch.stack(log_probs)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones).to(self.device)

        # states = torch.tensor(states).to(self.device)
        # actions = torch.tensor(actions).to(self.device)
        # rewards = torch.tensor(rewards).to(self.device)
        # returns = torch.tensor(returns).to(self.device)
        # o_log_probs = torch.tensor(log_probs).to(self.device)
        # next_states = torch.tensor(next_states).to(self.device)
        # dones = torch.tensor(dones).to(self.device)

        for state, action, reward, rets, probs, next_state, done in self.get_minibatches(states, actions, rewards, returns, o_log_probs, next_states, dones):
            # Compute value targets
            values = self.value(state).squeeze()
            # next_values = self.value(next_state).squeeze()
            # advantages, returns = self.compute_gae(reward, values, next_values, done, self.gamma, self.lam)
            
            # Update value function
            advantages = rets - values
            value_loss = (advantages ** 2).mean()
            # self.optim_value.zero_grad()
            # value_loss.backward()
            # self.optim_value.step()
            
            # Compute policy ratio
            log_probs = self.log_prob(state, action)
            # print("probs", state.shape, action.shape, probs.shape, log_probs.shape)
            ratio = torch.exp(log_probs - probs)
            
            # Compute surrogate objective
            policy_loss_1 = ratio * advantages
            policy_loss_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            loss = policy_loss + value_loss
            
            # Update policy
            self.optim_policy.zero_grad()
            loss.backward()
            self.optim_policy.step()
