import numpy as np
import gymnasium as gym
import loco_mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
import cv2
import os
import math
from collections import deque

# Create a custom wrapper to rotate the rendered frames
class RotatedRenderWrapper(gym.Wrapper):
    def render(self):
        # Get the original frame from the environment
        frame = self.env.render()
        # Rotate 90 degrees clockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated_frame

# Improved DeepMimic-style wrapper with proper reference state initialization and curriculum learning
class DeepMimicRewardWrapper(gym.Wrapper):
    def __init__(self, env, reference_data, state_mean, state_std, action_mean, action_std, 
                 w_p=0.5, w_v=0.5, w_e=0.0, w_tc=0.0):
        super(DeepMimicRewardWrapper, self).__init__(env)
        self.reference_data = reference_data
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        
        # Weights for different reward components
        self.w_p = w_p  # pose error weight - reduced from 0.65
        self.w_v = w_v  # velocity error weight - increased from 0.3
        self.w_e = w_e  # end-effector error weight
        self.w_tc = w_tc  # task-specific weight
        
        self.current_idx = 0
        self.max_idx = len(reference_data['states']) - 1
        
        # Determine position and velocity indices
        self.state_dim = reference_data['states'].shape[1]
        self.pos_indices = list(range(0, self.state_dim // 2))
        self.vel_indices = list(range(self.state_dim // 2, self.state_dim))
        
        # Curriculum learning for early termination
        self.base_pose_error_threshold = 5.0
        self.max_pose_error_threshold = 10.0
        self.curr_pose_error_threshold = self.base_pose_error_threshold
        self.early_terminated = False
        
        # Training progress tracking
        self.total_updates = 0
        self.max_updates = 500
        self.episode_count = 0
        self.total_rewards = []
        
        print(f"State dimensions: {self.state_dim}")
        print(f"Position indices: {len(self.pos_indices)} indices")
        print(f"Velocity indices: {len(self.vel_indices)} indices")
        
    def reset(self, **kwargs):
        # CRITICAL FIX: Random Reference State Initialization (RSI)
        # Sample a random starting point in the reference motion
        self.current_idx = np.random.randint(0, self.max_idx)
        self.early_terminated = False
        self.episode_count += 1
        
        # Update early termination threshold based on curriculum progress
        progress = min(self.total_updates / self.max_updates, 1.0)
        self.curr_pose_error_threshold = self.base_pose_error_threshold + progress * (self.max_pose_error_threshold - self.base_pose_error_threshold)
        
        obs, info = super().reset(**kwargs)
        
        # Debug info
        if self.episode_count % 10 == 0:
            print(f"Episode {self.episode_count}: Starting from reference frame {self.current_idx}")
            print(f"Current pose error threshold: {self.curr_pose_error_threshold:.4f}")
            if hasattr(self, 'total_rewards') and len(self.total_rewards) > 0:
                print(f"Recent episode rewards: min={min(self.total_rewards[-10:]):.4f}, "
                      f"max={max(self.total_rewards[-10:]):.4f}, "
                      f"avg={sum(self.total_rewards[-10:])/len(self.total_rewards[-10:]):.4f}")
        
        # Add reference state to info
        ref_state = self.reference_data['states'][self.current_idx]
        info['reference_state'] = ref_state
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get reference state and action for the current timestep
        ref_state = self.reference_data['states'][min(self.current_idx, self.max_idx)]
        
        # Normalize current state and reference state
        norm_obs = (obs - self.state_mean) / self.state_std
        norm_ref_state = (ref_state - self.state_mean) / self.state_std
        
        # CRITICAL FIX: Separate position and velocity components for reward calculation
        # Position error (first half of state vector)
        pos_error = np.mean(np.square(
            norm_obs[self.pos_indices] - norm_ref_state[self.pos_indices]
        ))
        
        # Velocity error (second half of state vector)
        vel_error = np.mean(np.square(
            norm_obs[self.vel_indices] - norm_ref_state[self.vel_indices]
        ))
        
        # CRITICAL FIX: Use less aggressive dampening for better learning signal
        pose_reward = math.exp(-0.8 * pos_error)  # Less aggressive dampening
        velocity_reward = math.exp(-0.001 * vel_error)  # Much more lenient for velocities
        
        # Calculate end-effector reward (simplified for now)
        end_effector_reward = 1.0
        
        # Combine rewards using weights
        imitation_reward = (
            self.w_p * pose_reward + 
            self.w_v * velocity_reward + 
            self.w_e * end_effector_reward
        )
        
        # Add task reward if needed
        task_reward = self.w_tc * reward  # original environment reward
        combined_reward = imitation_reward + task_reward
        
        # CRITICAL FIX: Early termination check based on position error only
        # Implement warmup period and curriculum-based threshold
        if self.total_updates >= 50:  # No early termination during initial learning
            if not self.early_terminated and pos_error > self.curr_pose_error_threshold:
                # Don't actually terminate the episode, just mark it as terminated
                # and zero out future rewards (DeepMimic approach)
                self.early_terminated = True
                if self.episode_count % 10 == 0:
                    print(f"Early termination at frame {self.current_idx}. Position error: {pos_error:.4f}")
        
        # If early terminated, set reward to 0
        if self.early_terminated:
            combined_reward = 0.0
        
        # Increment reference frame
        self.current_idx += 1
        
        # Add extra info for debugging
        info['pose_reward'] = pose_reward
        info['velocity_reward'] = velocity_reward
        info['position_error'] = pos_error
        info['velocity_error'] = vel_error
        info['reference_frame'] = self.current_idx
        info['early_terminated'] = self.early_terminated
        
        return obs, combined_reward, terminated, truncated, info
    
    def set_update_counter(self, update_count):
        # Allow the PPO trainer to tell the environment how many updates have occurred
        self.total_updates = update_count

# Create needed directories
video_folder = "videos"
models_folder = "models"
logs_folder = "logs"

for folder in [video_folder, models_folder, logs_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print(f"loco_mujoco version: {loco_mujoco.__version__}")

# -----------------------------------------------------------------------------
# Create the environment with the perfect dataset
# -----------------------------------------------------------------------------
env = gym.make(
    "LocoMujoco",
    env_name="HumanoidTorque.run",
    dataset_type="perfect",
    render_mode="rgb_array"
)

# -----------------------------------------------------------------------------
# Create dataset from the underlying loco_mujoco environment.
# -----------------------------------------------------------------------------
dataset = env.unwrapped.create_dataset()

# Print dataset info for debugging
print("Dataset keys:", dataset.keys())
states = dataset['states']
actions = dataset['actions']
print("States shape:", states.shape)
print("Actions shape:", actions.shape)
print("State range: min =", np.min(states), "max =", np.max(states))
print("Action range: min =", np.min(actions), "max =", np.max(actions))

# Normalize states and actions for better training
state_mean = np.mean(states, axis=0)
state_std = np.std(states, axis=0) + 1e-8  # to avoid division by zero
action_mean = np.mean(actions, axis=0)
action_std = np.std(actions, axis=0) + 1e-8

# Analyze reference motion for better understanding
print("\nAnalyzing reference motion:")
state_dim = states.shape[1]
pos_dim = state_dim // 2
position_data = states[:, :pos_dim]
velocity_data = states[:, pos_dim:]
print(f"Position component range: min={np.min(position_data):.4f}, max={np.max(position_data):.4f}")
print(f"Velocity component range: min={np.min(velocity_data):.4f}, max={np.max(velocity_data):.4f}")
pos_std = np.std(position_data)
vel_std = np.std(velocity_data)
print(f"Position std: {pos_std:.4f}, Velocity std: {vel_std:.4f}, Ratio: {vel_std/pos_std:.4f}x")

# Now wrap the environment with DeepMimic rewards
env = DeepMimicRewardWrapper(env, dataset, state_mean, state_std, action_mean, action_std)

# Wrap the environment to rotate frames before recording
env = RotatedRenderWrapper(env)

# Wrap with RecordVideo to automatically record episodes
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep_id: ep_id % 50 == 0)

# -----------------------------------------------------------------------------
# Define PPO network architecture (improved version)
# -----------------------------------------------------------------------------
state_dim = states.shape[1]
action_dim = actions.shape[1]

# PPO hyperparameters
gamma = 0.995  # Increased discount factor for longer-term credit assignment
lam = 0.95    # GAE lambda parameter
eps_clip = 0.2  # PPO clip parameter
ppo_epochs = 10  # PPO epochs per update
mini_batch_size = 64  # Mini batch size
value_coef = 0.5  # Value loss coefficient
entropy_coef = 0.015  # Slightly increased for more exploration
initial_lr = 3e-4  # Learning rate

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor (wider network)
        self.features = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(),
            nn.Linear(512, 384),
            nn.ELU(),
            nn.Linear(384, 256),
            nn.ELU()
        )
        
        # Policy head (actor)
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        
        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                module.bias.data.zero_()
        
        # Actor output layer initialization
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        self.actor_mean.bias.data.zero_()
        
    def forward(self, x):
        features = self.features(x)
        
        # Actor (policy) output
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic (value) output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean.detach().numpy()[0]
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        # Calculate log probability for later use
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().numpy()[0], log_prob.detach().item()
    
    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        
        dist = Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action_log_probs, entropy, value

# Initialize PPO model
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=initial_lr, eps=1e-5)

# Create scheduler for learning rate decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

# -----------------------------------------------------------------------------
# PPO training functions (with fixes)
# -----------------------------------------------------------------------------
def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
        
    return returns

def ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, 
               clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
    # Convert to tensor
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    old_log_probs = torch.FloatTensor(log_probs).unsqueeze(1)  # Fix dimension
    returns = torch.FloatTensor(returns).unsqueeze(1)  # Fix dimension
    advantages = torch.FloatTensor(advantages)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.unsqueeze(1)  # Fix dimension
    
    # Tracking metrics
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    # PPO update for specified number of epochs
    for _ in range(ppo_epochs):
        # Create mini-batches
        batch_size = len(states)
        indices = np.random.permutation(batch_size)
        
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            idx = indices[start_idx:end_idx]
            
            if len(idx) == 0:
                continue  # Skip empty batches
                
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_returns = returns[idx]
            batch_advantages = advantages[idx]
            
            # Get current policy evaluation
            new_log_probs, entropy, values = model.evaluate(batch_states, batch_actions)
            
            # Calculate the ratio (policy / old_policy)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # Calculate surrogate losses (PPO's clipped objective)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
            
            # Calculate actor, critic, and entropy losses
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # CRITICAL FIX: Correct the dimensions in MSE loss
            critic_loss = F.mse_loss(values, batch_returns)
            
            entropy_loss = -entropy.mean()
            
            # Combined loss
            loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
            
            # Update the model
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients for stability
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Track metrics
            total_policy_loss += actor_loss.item()
            total_value_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
    
    # Return average losses
    num_updates = ppo_epochs * (len(states) // mini_batch_size)
    if num_updates == 0:
        return 0, 0, 0
    
    return (total_policy_loss / num_updates, 
            total_value_loss / num_updates, 
            total_entropy / num_updates)

# -----------------------------------------------------------------------------
# Main training loop with improvements
# -----------------------------------------------------------------------------
print("Starting DeepMimic-style PPO training...")

# Training parameters
num_episodes = 1000  # Increased for better learning
max_steps_per_episode = 1000
max_total_steps = 500000  # Cap total training steps
eval_interval = 50
update_interval = 2048  # PPO update after this many steps

# Storage for combined episodes
all_states = []
all_actions = []
all_log_probs = []
all_rewards = []
all_values = []
all_masks = []

total_steps = 0
episode = 0
update_number = 0

# For tracking training progress
episode_rewards = []
episode_lengths = []
early_term_count = 0
best_eval_reward = -float('inf')

# Buffer for storing experience
buffer_states = []
buffer_actions = []
buffer_log_probs = []
buffer_rewards = []
buffer_values = []
buffer_masks = []

# Main training loop
while total_steps < max_total_steps and episode < num_episodes:
    episode += 1
    obs, info = env.reset()
    done = False
    step = 0
    episode_reward = 0
    
    # Storage for current episode
    episode_states = []
    episode_actions = []
    episode_log_probs = []
    episode_rewards = []
    episode_values = []
    episode_masks = []
    
    # Episode execution loop
    while not done and step < max_steps_per_episode:
        # Get normalized state
        norm_state = (obs - state_mean) / state_std
        
        # Select action from policy
        action, log_prob = model.get_action(norm_state)
        
        # Log the pre-execution data
        with torch.no_grad():
            # Calculate value for the current state
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0)
            _, _, value = model.forward(state_tensor)
            value_item = value.item()
        
        # Denormalize action for the environment
        denorm_action = action * action_std + action_mean
        
        # Execute action in the environment
        next_obs, reward, terminated, truncated, info = env.step(denorm_action)
        done = terminated or truncated
        
        # For debugging: print info about rewards
        if step % 100 == 0:
            pose_reward = info.get('pose_reward', 0)
            vel_reward = info.get('velocity_reward', 0)
            pos_error = info.get('position_error', 0)
            vel_error = info.get('velocity_error', 0)
            print(f"Step {step} - Pose: {pose_reward:.4f}, Vel: {vel_reward:.4f}, "
                  f"PosErr: {pos_error:.4f}, VelErr: {vel_error:.4f}, "
                  f"Reward: {reward:.4f}")
        
        # Check early termination
        if info.get('early_terminated', False):
            early_term_count += 1
        
        # Store episode data
        episode_states.append(norm_state)
        episode_actions.append(action)  # Store normalized action
        episode_rewards.append(reward)
        episode_log_probs.append(log_prob)
        episode_values.append(value_item)
        episode_masks.append(1 - done)
        
        # Also add to buffer
        buffer_states.append(norm_state)
        buffer_actions.append(action)
        buffer_log_probs.append(log_prob)
        buffer_rewards.append(reward)
        buffer_values.append(value_item)
        buffer_masks.append(1 - done)
        
        # Update for next iteration
        obs = next_obs
        episode_reward += reward
        step += 1
        total_steps += 1
        
        # Check if we need to update the policy
        if len(buffer_states) >= update_interval:
            with torch.no_grad():
                # Get value for the final state
                next_state = (obs - state_mean) / state_std
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, _, next_value = model.forward(next_state_tensor)
                next_value_item = next_value.item()
            
            # Compute advantage estimates
            returns = compute_gae(next_value_item, buffer_rewards, buffer_masks, 
                                 buffer_values, gamma, lam)
            advantages = np.array(returns) - np.array(buffer_values)
            
            # Update the policy
            policy_loss, value_loss, entropy = ppo_update(
                model, optimizer, buffer_states, buffer_actions, buffer_log_probs, 
                returns, advantages, eps_clip, value_coef, entropy_coef
            )
            
            # Decay learning rate
            scheduler.step()
            
            # Clear the buffer
            buffer_states = []
            buffer_actions = []
            buffer_log_probs = []
            buffer_rewards = []
            buffer_values = []
            buffer_masks = []
            
            update_number += 1
            
            # Update the environment's curriculum
            if hasattr(env.unwrapped, 'set_update_counter'):
                env.unwrapped.set_update_counter(update_number)
            
            print(f"Update {update_number} - Policy Loss: {policy_loss:.6f}, "
                  f"Value Loss: {value_loss:.6f}, Entropy: {entropy:.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save model every 100 updates
            if update_number % 100 == 0:
                torch.save(model.state_dict(), f"{models_folder}/deepmimic_update_{update_number}.pt")
                
                # Save training stats
                np.save(f"{logs_folder}/rewards_{update_number}.npy", np.array(episode_rewards))
                np.save(f"{logs_folder}/lengths_{update_number}.npy", np.array(episode_lengths))
                
                # Plot progress if matplotlib is available
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    plt.subplot(2, 1, 1)
                    plt.plot(episode_rewards)
                    plt.title('Episode Rewards')
                    plt.grid(True)
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(episode_lengths)
                    plt.title('Episode Lengths')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(f"{logs_folder}/training_progress_{update_number}.png")
                    plt.close()
                except ImportError:
                    print("Matplotlib not available for plotting")
    
    # Calculate final returns for this episode
    with torch.no_grad():
        # Get value for the final state if not done
        if not done:
            next_state = (obs - state_mean) / state_std
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, _, next_value = model.forward(next_state_tensor)
            next_value_item = next_value.item()
        else:
            next_value_item = 0.0
    
    # Compute advantage estimates for this episode
    episode_returns = compute_gae(next_value_item, episode_rewards, episode_masks, 
                                 episode_values, gamma, lam)
    
    # Log episode stats
    episode_length = step
    episode_lengths.append(episode_length)
    episode_rewards.append(episode_reward)
    
    if hasattr(env.unwrapped, 'total_rewards'):
        env.unwrapped.total_rewards.append(episode_reward)
    
    print(f"Episode {episode}/{num_episodes} - Steps: {episode_length} - Reward: {episode_reward:.4f}")
    
    # Evaluate the policy periodically
    if episode % eval_interval == 0:
        print(f"\nEvaluation after episode {episode}:")
        print(f"Average reward over last {min(len(episode_rewards), 10)} episodes: "
              f"{sum(episode_rewards[-10:]) / min(len(episode_rewards), 10):.4f}")
        print(f"Average episode length: {sum(episode_lengths[-10:]) / min(len(episode_lengths), 10):.1f}")
        print(f"Early terminations: {early_term_count} / {episode}")
        
        # Create a separate environment for evaluation
        eval_env = DeepMimicRewardWrapper(
            gym.make("LocoMujoco", env_name="HumanoidTorque.run", dataset_type="perfect", render_mode="rgb_array"),
            dataset, state_mean, state_std, action_mean, action_std
        )
        eval_env = RotatedRenderWrapper(eval_env)
        eval_env = gym.wrappers.RecordVideo(
            eval_env, 
            video_folder=f"{video_folder}/eval_ep{episode}", 
            episode_trigger=lambda ep_id: True
        )
        
        # Run evaluation
        eval_obs, _ = eval_env.reset()
        eval_done = False
        eval_steps = 0
        eval_reward = 0
        
        while not eval_done and eval_steps < 500:
            # Get normalized observation
            eval_norm_obs = (eval_obs - state_mean) / state_std
            
            # Get deterministic action
            with torch.no_grad():
                eval_action = model.get_action(eval_norm_obs, deterministic=True)[0]
            
            # Denormalize action
            eval_denorm_action = eval_action * action_std + action_mean
            
            # Step environment
            eval_obs, eval_rew, eval_term, eval_trunc, _ = eval_env.step(eval_denorm_action)
            eval_done = eval_term or eval_trunc
            eval_reward += eval_rew
            eval_steps += 1
        
        print(f"Evaluation - Steps: {eval_steps}, Reward: {eval_reward:.4f}")
        eval_env.close()
        
        # Save if this is the best model so far
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            torch.save(model.state_dict(), f"{models_folder}/deepmimic_best.pt")
            print(f"New best model saved with reward: {best_eval_reward:.4f}")

# Save the final model
torch.save(model.state_dict(), f"{models_folder}/deepmimic_final.pt")

# -----------------------------------------------------------------------------
# Final evaluation of the learned policy
# -----------------------------------------------------------------------------
model.eval()

# Create a clean environment for final evaluation
final_env = DeepMimicRewardWrapper(
    gym.make("LocoMujoco", env_name="HumanoidTorque.run", dataset_type="perfect", render_mode="rgb_array"),
    dataset, state_mean, state_std, action_mean, action_std
)
final_env = RotatedRenderWrapper(final_env)
final_env = gym.wrappers.RecordVideo(final_env, video_folder=f"{video_folder}/final", 
                                      episode_trigger=lambda ep_id: True)

# Reset the environment for evaluation
obs, info = final_env.reset()
total_reward = 0
done = False
step_count = 0
print("Final policy evaluation...")

try:
    while not done and step_count < 1000:  # safeguard limit
        # Normalize observation
        norm_obs = (obs - state_mean) / state_std
        
        # Get action (deterministically for evaluation)
        with torch.no_grad():
            action = model.get_action(norm_obs, deterministic=True)[0]
            
        # Denormalize action for the environment
        denorm_action = action * action_std + action_mean
        
        # Step the environment
        obs, reward, terminated, truncated, info = final_env.step(denorm_action)
        total_reward += reward
        done = terminated or truncated
        
        # Get the current rendered frame (already rotated by the wrapper)
        frame = final_env.render()
        
        # Scale down to 640p height while maintaining aspect ratio
        height, width = frame.shape[:2]
        new_height = 640
        new_width = int(width * (new_height / height))
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Display the resized frame using OpenCV
        cv2.imshow("Final Evaluation", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        step_count += 1
        time.sleep(0.01)  # Short delay for visualization
        
except Exception as e:
    print(f"Error during evaluation: {e}")

print(f"Final evaluation finished after {step_count} steps with total reward: {total_reward:.4f}")
print(f"Videos saved to {video_folder}")
print(f"Models saved to {models_folder}")
final_env.close()
cv2.destroyAllWindows()
