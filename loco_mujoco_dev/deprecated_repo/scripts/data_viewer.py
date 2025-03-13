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

# Get base timestep
base_timestep = env._model.opt.timestep
print(f"Base timestep: {base_timestep} seconds")

# Get observation structure
obs_keys = env.get_all_observation_keys()
print("\nObservation keys from environment:")
print(obs_keys)

# Try to get observation space
observation_space = env._get_observation_space()
print(f"Observation space: {observation_space}")

# Handle observation space as a tuple (low, high)
if isinstance(observation_space, tuple) and len(observation_space) == 2:
    low_bounds, high_bounds = observation_space
    print(f"\nObservation space dimension: {len(low_bounds)}")
    
    # Print first few bounds for readability
    print(f"First 5 low bounds: {low_bounds[:5]}")
    print(f"First 5 high bounds: {high_bounds[:5]}")

# Extract action information from dataset
if 'actions' in data:
    actions_data = data["actions"]
    
    print("\nAction Space Information (from dataset):")
    print(f"Actions shape: {actions_data.shape}")
    print(f"Action dimension: {actions_data.shape[1]}")
    
    # Print sample actions
    print(f"\nSample actions:")
    print(f"First action: {actions_data[0]}")
    print(f"Middle action: {actions_data[len(actions_data)//2]}")
    print(f"Last action: {actions_data[-1]}")
    
    # Calculate bounds
    action_min = np.min(actions_data, axis=0)
    action_max = np.max(actions_data, axis=0)
    action_mean = np.mean(actions_data, axis=0)
    action_std = np.std(actions_data, axis=0)
    
    print("\nAction statistics:")
    print(f"Minimum values: {action_min}")
    print(f"Maximum values: {action_max}")
    print(f"Mean values: {action_mean}")
    print(f"Standard deviation: {action_std}")
    
    # Create a Box space equivalent for the actions
    print("\nEquivalent Gym Box space would be:")
    print(f"Box(low={action_min.tolist()}, high={action_max.tolist()}, shape=({actions_data.shape[1]},), dtype=np.float32)")
    
    # Additional information about temporal aspects
    print("\nTemporal information:")
    print(f"Number of timesteps: {len(actions_data)}")
    if hasattr(env, "dt"):
        print(f"Environment dt: {env.dt}")
        print(f"Total trajectory duration: {len(actions_data) * env.dt} seconds")
    else:
        effective_dt = base_timestep * getattr(env, 'frame_skip', 1)
        print(f"Estimated effective dt: {effective_dt} seconds")
        print(f"Estimated trajectory duration: {len(actions_data) * effective_dt} seconds")
    
    # Check for zero actions
    zero_actions = np.all(actions_data == 0, axis=1).sum()
    print(f"\nNumber of all-zero actions: {zero_actions} ({zero_actions/len(actions_data)*100:.2f}%)")
    
    # Check for action continuity
    action_diffs = np.abs(actions_data[1:] - actions_data[:-1])
    mean_diff = np.mean(action_diffs)
    max_diff = np.max(action_diffs)
    print(f"Mean absolute change between consecutive actions: {mean_diff}")
    print(f"Maximum absolute change between consecutive actions: {max_diff}")
else:
    print("\nNo 'actions' key found in the dataset.")

# Try to get state information too
if 'states' in data:
    states = data["states"]
    print(f"\nStates shape: {states.shape}")
    print(f"First state example: {states[0][:5]}...")  # Just first 5 values

# Print action-observation mapping (which action affects which observation)
try:
    print("\nMapping actuators to joints:")
    
    # Get names of joints that are actuated
    actuator_names = []
    for i in range(env._model.nu):
        # Get actuator name
        name_adr = env._model.name_actuatoradr[i]
        length = 0
        # Find null terminator to get string length
        while env._model.names[name_adr + length] != 0:
            length += 1
        name = bytes(env._model.names[name_adr:name_adr+length]).decode('utf-8')
        actuator_names.append(name)
    
    print(f"Actuator names: {actuator_names}")
except Exception as e:
    print(f"Error extracting actuator names: {e}")

print("\nScript execution completed successfully!")
