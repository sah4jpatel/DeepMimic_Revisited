import numpy as np
import mujoco
import matplotlib.pyplot as plt
import seaborn as sns
from loco_mujoco.environments.humanoids import HumanoidTorque
import time
import os

# Path to the dataset
dataset_path = "/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"

print("\n===== DATASET STRUCTURE ANALYSIS =====")
# Load and inspect the dataset structure
data = np.load(dataset_path)
print(f"Dataset path: {dataset_path}")
print(f"Dataset keys: {list(data.keys())}")

# Print detailed information about each key in the dataset
for key in data.keys():
    array = data[key]
    print(f"\n{key} array details:")
    print(f"  Shape: {array.shape}")
    print(f"  Data type: {array.dtype}")
    if array.size > 0:
        print(f"  Min value: {np.min(array)}")
        print(f"  Max value: {np.max(array)}")
        print(f"  Mean value: {np.mean(array):.6f}")
        print(f"  Standard deviation: {np.std(array):.6f}")
    
    # Print sample values based on dimensionality
    if array.ndim == 1:
        print(f"  Sample values (first 10): {array[:10]}")
    elif array.ndim == 2:
        print(f"  First row (first 10 values): {array[0][:10]}")
        print(f"  Middle row (first 10 values): {array[len(array)//2][:10]}")
        print(f"  Last row (first 10 values): {array[-1][:10]}")
    else:
        print(f"  Higher dimensional array - samples omitted")

print("\n===== ENVIRONMENT CONFIGURATION =====")
# Create environment with random_start=False
env = HumanoidTorque(use_foot_forces=True, random_start=False)

# Get the base timestep from the model
base_timestep = env._model.opt.timestep
print(f"Base timestep: {base_timestep} seconds")

# Print detailed model information
print(f"\nModel details:")
print(f"  Model name: {env._model.name}")
print(f"  Number of degrees of freedom (nv): {env._model.nv}")
print(f"  Number of position coordinates (nq): {env._model.nq}")
print(f"  Number of bodies: {env._model.nbody}")
print(f"  Number of joints: {env._model.njnt}")
print(f"  Number of actuators/control dimensions: {env._model.nu}")
print(f"  Number of tendons: {env._model.ntendon}")
print(f"  Number of sensors: {env._model.nsensor}")

# Get observation and action space details
print("\nObservation space details:")
observation_space = env.observation_space
print(f"  Shape: {observation_space.shape}")
print(f"  Low bounds (first 10): {observation_space.low[:10]}")
print(f"  High bounds (first 10): {observation_space.high[:10]}")

print("\nAction space details:")
action_space = env.action_space
print(f"  Shape: {action_space.shape}")
print(f"  Low bounds: {action_space.low}")
print(f"  High bounds: {action_space.high}")

# Get observation structure with more detailed information
obs_keys = env.get_all_observation_keys()
print("\nDetailed observation structure:")
for i, key in enumerate(obs_keys):
    print(f"  Dimension {i}: {key}")

# Extract joint and actuator names if available
try:
    print("\nJoint details:")
    for i in range(env._model.njnt):
        joint_name = env._model.joint(i).name
        print(f"  Joint {i}: {joint_name}")
        
    print("\nActuator details:")
    for i in range(env._model.nu):
        actuator_name = env._model.actuator(i).name
        print(f"  Actuator {i}: {actuator_name}")
except Exception as e:
    print(f"Could not retrieve joint/actuator names: {e}")

# Reset environment and capture initial state
initial_obs = env.reset()
print("\nInitial observation after reset:")
print(f"  Shape: {initial_obs.shape}")
print(f"  Values (first 15): {initial_obs[:15]}")
print(f"  qpos (first 10): {env._data.qpos[:10]}")
print(f"  qvel (first 10): {env._data.qvel[:10]}")

print("\n===== STATE AND ACTION ANALYSIS =====")
# Extract and analyze states from dataset
states = data["states"]
print(f"States analysis:")
print(f"  Shape: {states.shape} - {states.shape[0]} timesteps with {states.shape[1]} dimensions per state")
print(f"  Value range: Min={np.min(states):.6f}, Max={np.max(states):.6f}")

# Analyze state dimensions
state_mins = np.min(states, axis=0)
state_maxs = np.max(states, axis=0)
state_means = np.mean(states, axis=0)
state_stds = np.std(states, axis=0)

print("\nPer-dimension state statistics (first 20 dimensions):")
print("  Dim | Min Value | Max Value | Mean | Std Dev")
print("  --- | --------- | --------- | ---- | -------")
for i in range(min(20, states.shape[1])):
    print(f"  {i:3d} | {state_mins[i]:9.4f} | {state_maxs[i]:9.4f} | {state_means[i]:6.4f} | {state_stds[i]:7.4f}")

# Check for actions in the dataset
if "actions" in data:
    actions = data["actions"]
    print(f"\nActions analysis:")
    print(f"  Shape: {actions.shape} - {actions.shape[1]} action dimensions")
    print(f"  Value range: Min={np.min(actions):.6f}, Max={np.max(actions):.6f}")
    
    # Analyze action dimensions
    action_mins = np.min(actions, axis=0)
    action_maxs = np.max(actions, axis=0)
    action_means = np.mean(actions, axis=0)
    action_stds = np.std(actions, axis=0)
    
    print("\nPer-dimension action statistics:")
    print("  Dim | Min Value | Max Value | Mean | Std Dev")
    print("  --- | --------- | --------- | ---- | -------")
    for i in range(actions.shape[1]):
        print(f"  {i:3d} | {action_mins[i]:9.4f} | {action_maxs[i]:9.4f} | {action_means[i]:6.4f} | {action_stds[i]:7.4f}")

# Analyze rewards if available
if "rewards" in data:
    rewards = data["rewards"]
    print(f"\nRewards analysis:")
    print(f"  Shape: {rewards.shape}")
    print(f"  Statistics: Min={np.min(rewards):.6f}, Max={np.max(rewards):.6f}, Mean={np.mean(rewards):.6f}, Std={np.std(rewards):.6f}")
    print(f"  Sample values (first 20): {rewards[:20]}")
    
    # Histogram of rewards (text-based or save to file)
    bins = np.linspace(np.min(rewards), np.max(rewards), 10)
    hist, edges = np.histogram(rewards, bins=bins)
    print("\nReward distribution histogram:")
    for i in range(len(hist)):
        print(f"  {edges[i]:.4f} to {edges[i+1]:.4f}: {hist[i]} values")

# Analyze episode information if available
if "dones" in data:
    dones = data["dones"]
    print(f"\nEpisode analysis:")
    print(f"  Shape: {dones.shape}")
    print(f"  Number of episode endings: {np.sum(dones)}")
    
    # Calculate episode lengths
    episode_ends = np.where(dones)[0]
    if len(episode_ends) > 0:
        episode_lengths = np.diff(np.concatenate([[0], episode_ends + 1]))
        print(f"  Episode statistics: Count={len(episode_lengths)}, Mean length={np.mean(episode_lengths):.2f}")
        print(f"  Episode length range: Min={np.min(episode_lengths)}, Max={np.max(episode_lengths)}")
        print(f"  Sample episode lengths: {episode_lengths[:10]}")

print("\n===== DATA VISUALIZATION =====")
# Create directory for saving plots
plot_dir = "humanoid_data_analysis"
os.makedirs(plot_dir, exist_ok=True)

# State visualization functions
def visualize_state_distributions(states, save_path):
    """Generate histograms of state distributions for key dimensions"""
    n_dims = min(16, states.shape[1])
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        ax.hist(states[:, i], bins=50, alpha=0.7)
        ax.set_title(f'State Dimension {i} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved state distribution histograms to {save_path}")

def visualize_state_trajectories(states, save_path):
    """Plot state trajectories over time for key dimensions"""
    n_dims = min(16, states.shape[1])
    n_steps = min(1000, states.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        ax.plot(range(n_steps), states[:n_steps, i])
        ax.set_title(f'State Dimension {i} Trajectory')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved state trajectory plots to {save_path}")

def visualize_state_correlations(states, save_path):
    """Generate correlation matrix for state dimensions"""
    n_dims = min(20, states.shape[1])
    
    # Sample states if there are too many
    if states.shape[0] > 5000:
        indices = np.random.choice(states.shape[0], 5000, replace=False)
        state_sample = states[indices, :n_dims]
    else:
        state_sample = states[:, :n_dims]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(state_sample.T)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('State Dimension Correlation Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved state correlation matrix to {save_path}")

# Generate visualization plots
print("Generating data visualizations:")
visualize_state_distributions(states, f"{plot_dir}/state_distributions.png")
visualize_state_trajectories(states, f"{plot_dir}/state_trajectories.png")
visualize_state_correlations(states, f"{plot_dir}/state_correlations.png")

# Analyze state-action correlations if actions are available
if "actions" in data:
    def visualize_state_action_correlation(states, actions, save_path):
        """Analyze correlation between states and actions"""
        n_state_dims = min(states.shape[1], 20)
        n_action_dims = actions.shape[1]
        
        # Create a combined matrix of states and actions
        if states.shape[0] > 5000:
            indices = np.random.choice(states.shape[0], 5000, replace=False)
            state_sample = states[indices, :n_state_dims]
            action_sample = actions[indices]
        else:
            state_sample = states[:, :n_state_dims]
            action_sample = actions
            
        combined = np.hstack([state_sample, action_sample])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(combined.T)
        
        # Extract the state-action correlation submatrix
        state_action_corr = corr_matrix[:n_state_dims, n_state_dims:]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(state_action_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('State-Action Correlation Matrix')
        plt.xlabel('Action Dimensions')
        plt.ylabel('State Dimensions')
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved state-action correlation matrix to {save_path}")
    
    visualize_state_action_correlation(states, actions, f"{plot_dir}/state_action_correlation.png")

print("\n===== ENHANCED STATE VISUALIZATION =====")
# Function to visualize a sequence of states with detailed logging
def visualize_states(states, env, num_frames=500, log_interval=100):
    """Visualize states from the dataset with detailed information logging"""
    # Reset environment
    env.reset()
    
    print(f"Visualizing {min(len(states), num_frames)} frames from the dataset...")
    
    # Prepare data collection arrays
    logged_data = {
        'frame': [],
        'qpos': [],
        'qvel': [],
        'state_vector': [],
        'com_pos': [],
        'com_vel': []
    }
    
    if "actions" in data:
        logged_data['action'] = []
    
    if "rewards" in data:
        logged_data['reward'] = []
    
    # Loop through states
    for i in range(min(len(states), num_frames)):
        # Set environment state from the dataset state
        full_obs = np.zeros(len(obs_keys))
        full_obs[2:] = states[i]  # Skip the first two values (pelvis x,y which can be 0)
        
        # Set state - must be in same order as observation keys
        env.set_sim_state(full_obs)
        
        # Gather detailed state information
        current_qpos = env._data.qpos.copy()
        current_qvel = env._data.qvel.copy()
        
        # Try to get center of mass information
        try:
            com_pos = np.zeros(3)
            com_vel = np.zeros(3)
            mujoco.mj_getCOMpos(env._model, env._data, com_pos)
            mujoco.mj_getCOMvel(env._model, env._data, com_vel, com_vel)
        except:
            com_pos = np.array([0, 0, 0])
            com_vel = np.array([0, 0, 0])
        
        # Log data at specified intervals and for the first and last frames
        if i % log_interval == 0 or i == 0 or i == min(len(states), num_frames) - 1:
            logged_data['frame'].append(i)
            logged_data['qpos'].append(current_qpos.copy())
            logged_data['qvel'].append(current_qvel.copy())
            logged_data['state_vector'].append(states[i].copy())
            logged_data['com_pos'].append(com_pos.copy())
            logged_data['com_vel'].append(com_vel.copy())
            
            if "actions" in data and i < len(data["actions"]):
                logged_data['action'].append(data["actions"][i].copy())
            
            if "rewards" in data and i < len(data["rewards"]):
                logged_data['reward'].append(data["rewards"][i])
            
            print(f"\nFrame {i} details:")
            print(f"  State vector (first 10): {states[i][:10]}")
            print(f"  qpos (first 10): {current_qpos[:10]}")
            print(f"  qvel (first 10): {current_qvel[:10]}")
            print(f"  Center of mass - pos: {com_pos}, vel: {com_vel}")
            
            if "actions" in data and i < len(data["actions"]):
                print(f"  Action: {data['actions'][i]}")
            
            if "rewards" in data and i < len(data["rewards"]):
                print(f"  Reward: {data['rewards'][i]}")
        
        # Render the current frame
        env.render()
        
        # Forward simulation
        mujoco.mj_forward(env._model, env._data)
        
    env.stop()
    print("Visualization complete")
    
    # Save logged data to numpy file
    log_data_path = f"{plot_dir}/visualization_data.npz"
    np.savez(log_data_path, **logged_data)
    print(f"Saved visualization data to {log_data_path}")
    
    return logged_data

# Visualize states
print("Starting visualization of motion sequence...")
visualization_data = visualize_states(states, env)


print("\n===== COMPREHENSIVE SUMMARY =====")
# Summary of all the information gathered
print("Environment Summary:")
print(f"  Model name: {env._model.name}")
print(f"  Simulation timestep: {base_timestep} seconds")
print(f"  Degrees of freedom: {env._model.nv}")
print(f"  Observation space: {observation_space.shape[0]} dimensions")
print(f"  Action space: {action_space.shape[0]} dimensions")

print("\nDataset Summary:")
print(f"  Dataset path: {dataset_path}")
print(f"  Dataset arrays: {list(data.keys())}")

if "states" in data:
    print(f"  State data: {data['states'].shape[0]} timesteps with {data['states'].shape[1]} dimensions")
    print(f"  State value range: {np.min(states):.4f} to {np.max(states):.4f}")

if "actions" in data:
    print(f"  Action data: {data['actions'].shape[0]} timesteps with {data['actions'].shape[1]} dimensions")
    print(f"  Action value range: {np.min(data['actions']):.4f} to {np.max(data['actions']):.4f}")

if "rewards" in data:
    print(f"  Reward data: {data['rewards'].shape[0]} values")
    print(f"  Reward range: {np.min(data['rewards']):.4f} to {np.max(data['rewards']):.4f}")

if "dones" in data:
    num_episodes = np.sum(data["dones"]) + 1
    print(f"  Episodes: {num_episodes}")
    if num_episodes > 1:
        episode_ends = np.where(data["dones"])[0]
        episode_lengths = np.diff(np.concatenate([[0], episode_ends + 1]))
        print(f"  Average episode length: {np.mean(episode_lengths):.2f} steps")

print("\nSuggested Next Steps:")
print("  1. Choose an ML model approach from the configurations discussed")
print("  2. Prepare and preprocess the data according to the recommendations")
print("  3. Implement and train the model with appropriate architecture")
print("  4. Evaluate model performance on held-out test data")
print("  5. Use the model for inference or further refinement")
