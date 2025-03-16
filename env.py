import gymnasium as gym
import mujoco
import mujoco_py
import numpy as np
import torch

class MuJoCoBackflipEnv(gym.Env):
    def __init__(self, xml, motion):
        super(MuJoCoBackflipEnv, self).__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = mujoco.MjViewer(self.sim)

        self.dt = self.model.opt.timestep  # MuJoCo timestep
        self.policy_freq = 1/30  # 30 Hz control frequency
        self.action_dim = self.model.nu  # Number of controllable joints
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel)  # Joint positions & velocities

        # self.ref_motion = self.load_reference_motion("backflip.npy")  # Load MoCap reference motion
        self.ref_motion = motion
        
        print("init", self.dt, self.policy_freq, self.action_dim, self.state_dim)
    
    def load_reference_motion(self, path):
        """Load reference motion for backflip."""
        return np.load(path)  # Array of joint angles over time

    def step(self, action):
        """Apply action and step the simulation."""
        self.apply_pd_control(action)
        for _ in range(int(self.policy_freq / self.dt)):  # Simulate at high frequency
            self.sim.step()

        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.check_termination()
        return obs, reward, done, {}

    def reset(self):
        """Reset simulation to a reference motion frame."""
        self.sim.reset()
        frame_idx = np.random.randint(0, len(self.ref_motion))  # RSI (Reference State Initialization)
        self.sim.data.qpos[:] = self.ref_motion[frame_idx][:len(self.sim.data.qpos)]
        return self.get_observation()

    def get_observation(self):
        """Return current joint angles and velocities."""
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel])

    def apply_pd_control(self, action):
        """Apply PD control to joints."""
        kp, kd = 100, 10  # PD gains
        for i in range(self.action_dim):
            torque = kp * (action[i] - self.sim.data.qpos[i]) - kd * self.sim.data.qvel[i]
            self.sim.data.ctrl[i] = torque  # Apply torque

    def compute_reward(self):
        """Imitation reward comparing current pose to reference motion."""
        ref_idx = min(len(self.ref_motion) - 1, int(self.sim.data.time * 30))  # Get closest reference frame
        ref_qpos = self.ref_motion[ref_idx][:len(self.sim.data.qpos)]

        # Pose similarity reward
        pose_diff = np.sum((self.sim.data.qpos - ref_qpos) ** 2)
        reward = np.exp(-2 * pose_diff)

        return reward

    def check_termination(self):
        """Terminate episode if character falls."""
        return self.sim.data.qpos[2] < 0.3  # Check if root height is too low

    def render(self):
        self.viewer.render()
