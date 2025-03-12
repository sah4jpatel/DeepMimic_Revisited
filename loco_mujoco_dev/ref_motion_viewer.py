import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import HumanoidTorque

# Path to the dataset
dataset_path = "/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"

# Load and inspect the dataset structure
data = np.load(dataset_path)
print("Dataset keys:", list(data.keys()))

# Create environment with random_start=False
env = HumanoidTorque(use_foot_forces=True, random_start=False)

# Get the base timestep from the model
base_timestep = env._model.opt.timestep

# Print the timestep
print(f"Base timestep: {base_timestep} seconds")


# Get observation structure
obs_keys = env.get_all_observation_keys()
print("\nObservation keys from environment:")
print(obs_keys)

# Extract states from dataset
states = data["states"]
print(f"\nStates shape: {states.shape}")
print(f"First state example: {states[0][:5]}...")  # Just first 5 values

# Function to visualize a sequence of states
def visualize_states(states, env, num_frames=500):
    # Reset environment
    env.reset()
    
    # Loop through states
    for i in range(min(len(states), num_frames)):
        # Set environment state from the dataset state
        # Note: First create a full observation for the environment
        full_obs = np.zeros(len(obs_keys))
        full_obs[2:] = states[i]  # Skip the first two values (pelvis x,y which can be 0)
        
        # Set state - must be in same order as observation keys
        env.set_sim_state(full_obs)
        
        # Render
        env.render()
        
        # Forward simulation for visualization only
        mujoco.mj_forward(env._model, env._data)
        
    env.stop()

# Visualize states
print("\nVisualizing motion sequence...")
visualize_states(states, env)
