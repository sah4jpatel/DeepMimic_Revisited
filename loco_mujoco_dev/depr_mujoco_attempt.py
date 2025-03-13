import os
import time
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer

class DeepMimicGymEnv(gym.Env):
    """Fixed environment with proper state handling"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ref_data_path, model_xml_path, debug_print=False):
        super(DeepMimicGymEnv, self).__init__()
        self.debug_print = debug_print

        # Load expert data
        data = np.load(ref_data_path, allow_pickle=True)
        self.ref_states = data['states']  # (T, 36)
        self.ref_actions = data['actions']  # (T, 13)
        self.T = len(self.ref_states)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Verify model dimensions
        self.qpos_size = self.model.nq  # Should be 37 (root(3) + joints(34))
        self.qvel_size = self.model.nv  # Should be 36
        
        # Observation space matches expert data (36 joints, excluding root)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(36,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                      shape=(13,), dtype=np.float32)
        
        self.viewer = None
        self.current_phase = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.current_phase = 0
        
        # Initialize from expert data (skip root position)
        expert_state = self.ref_states[0]
        self.data.qpos[3:3+36] = expert_state  # Assume first 3 are root XYZ
        self.data.qvel[:] = 0  # Start with zero velocity
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        """Get observation (joint positions only, excluding root)"""
        return self.data.qpos[3:39].copy().astype(np.float32)

    def step(self, action):
        # Map 13-dim action to first 13 actuators
        self.data.ctrl[:13] = action
        
        mujoco.mj_step(self.model, self.data)
        
        # Calculate reward against expert pose
        obs = self._get_obs()
        expert_pose = self.ref_states[self.current_phase]
        pose_diff = np.linalg.norm(obs - expert_pose)
        reward = np.exp(-0.5 * pose_diff**2)  # Smoother reward curve
        
        self.current_phase = (self.current_phase + 1) % self.T
        done = self.current_phase == 0
        
        return obs, reward, done, {}

    def render(self, speed=1.0):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()
        time.sleep(0.01 / speed)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

class BCPolicy(nn.Module):
    """Simplified policy network"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

def train_and_visualize():
    # Configuration
    ref_data_path = "/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"
    model_xml_path = "/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/environments/data/humanoid/humanoid_torque.xml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment and policy
    env = DeepMimicGymEnv(ref_data_path, model_xml_path, debug_print=True)
    policy = BCPolicy(36, 13).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    # Load expert data
    data = np.load(ref_data_path)
    expert_states = data['states']
    expert_actions = data['actions']

    # Training setup
    plt.ion()
    fig, ax = plt.subplots()
    losses = []
    
    # Training loop
    for epoch in range(100):
        # Training
        epoch_loss = 0
        for i in range(0, len(expert_states), 64):
            states = torch.FloatTensor(expert_states[i:i+64]).to(device)
            actions = torch.FloatTensor(expert_actions[i:i+64]).to(device)
            
            optimizer.zero_grad()
            pred_actions = policy(states)
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Store and plot loss
        avg_loss = epoch_loss / (len(expert_states)//64)
        losses.append(avg_loss)
        ax.clear()
        ax.plot(losses)
        plt.pause(0.01)
        
        # Visualization every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
            with torch.no_grad():
                state = env.reset()
                for _ in range(200):
                    action = policy(torch.FloatTensor(state).to(device)).cpu().numpy()
                    state, _, _, _ = env.step(action)
                    env.render(speed=4)

    # Final save and visualization
    torch.save(policy.state_dict(), "bc_policy.pth")
    print("Final visualization:")
    state = env.reset()
    for _ in range(500):
        action = policy(torch.FloatTensor(state).to(device)).cpu().numpy()
        state, _, _, _ = env.step(action)
        env.render(speed=1)
    
    plt.ioff()
    env.close()

if __name__ == "__main__":
    train_and_visualize()
