import numpy as np
import gymnasium as gym
import loco_mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import time
import cv2
import os

# Create a custom wrapper to rotate the rendered frames
class RotatedRenderWrapper(gym.Wrapper):
    def render(self):
        # Get the original frame from the environment
        frame = self.env.render()
        # Rotate 90 degrees clockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated_frame

# Create folder for video output if it doesn't exist
video_folder = "videos"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

print(f"loco_mujoco version: {loco_mujoco.__version__}")

# -----------------------------------------------------------------------------
# Create the environment with the perfect dataset and record video.
# -----------------------------------------------------------------------------
# First create the base environment
env = gym.make(
    "LocoMujoco",
    env_name="HumanoidTorque.run",
    dataset_type="perfect",
    render_mode="rgb_array"  # required for video recording
)

# Wrap the environment to rotate frames before recording
env = RotatedRenderWrapper(env)

# Wrap with RecordVideo to automatically record episodes
env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep_id: True)

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
print("Sample state:", states[0][:5], "...")
print("Sample action:", actions[0][:5], "...")

# -----------------------------------------------------------------------------
# Define a simple behavioral cloning policy using PyTorch
# -----------------------------------------------------------------------------
state_dim = states.shape[1]
action_dim = actions.shape[1]

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

policy = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)  # Lower learning rate for stable training
criterion = nn.MSELoss()

# -----------------------------------------------------------------------------
# Train the policy using behavioral cloning (imitation learning)
# -----------------------------------------------------------------------------
num_epochs = 50
batch_size = 64
num_samples = states.shape[0]

# Normalize states and actions for better training
state_mean = np.mean(states, axis=0)
state_std = np.std(states, axis=0) + 1e-8  # to avoid division by zero
action_mean = np.mean(actions, axis=0)
action_std = np.std(actions, axis=0) + 1e-8

states_norm = (states - state_mean) / state_std
actions_norm = (actions - action_mean) / action_std

print("Starting imitation learning training...")
for epoch in range(num_epochs):
    permutation = np.random.permutation(num_samples)
    epoch_loss = 0.0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        batch_states = torch.tensor(states_norm[indices], dtype=torch.float32)
        batch_actions = torch.tensor(actions_norm[indices], dtype=torch.float32)
        
        optimizer.zero_grad()
        predicted = policy(batch_states)
        loss = criterion(predicted, batch_actions)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
    epoch_loss /= num_batches  # Average loss per batch
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}")

# -----------------------------------------------------------------------------
# Evaluate the learned policy in the environment and record video
# -----------------------------------------------------------------------------
obs, info = env.reset()
total_reward = 0
done = False
step_count = 0
print("Running policy in the environment...")

try:
    while not done and step_count < 1000:  # safeguard limit
        # Normalize observation using training statistics
        obs_norm = (obs - state_mean) / state_std
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_norm = policy(obs_tensor).squeeze(0).numpy()
            # Denormalize the action
            action = action_norm * action_std + action_mean
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Get the current rendered frame (already rotated by the wrapper)
        frame = env.render()
        
        # Scale down to 640p height while maintaining aspect ratio
        # Note: After rotation, height and width are swapped
        height, width = frame.shape[:2]
        new_height = 640
        new_width = int(width * (new_height / height))
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Display the resized frame using OpenCV
        cv2.imshow("Simulation", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        step_count += 1
        time.sleep(0.01)  # Short delay for visualization
        
except Exception as e:
    print(f"Error during evaluation: {e}")

print(f"Episode finished after {step_count} steps with total reward: {total_reward:.4f}")
print(f"Video saved to {video_folder}")
env.close()
cv2.destroyAllWindows()
