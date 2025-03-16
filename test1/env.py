import gymnasium as gym
import mujoco
# import mujoco_py
import numpy as np
import torch
from utils import quaternion_difference

class MuJoCoBackflipEnv(gym.Env):
    def __init__(self, xml, motion):
        super(MuJoCoBackflipEnv, self).__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # self.model.opt.timestep = 0.001

        # self.sim = mujoco.MjSim(self.model)
        # self.viewer = mujoco.MjViewer(self.sim)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.dt = self.model.opt.timestep  # MuJoCo timestep
        self.policy_freq = 1/30  # 30 Hz control frequency
        self.action_dim = self.model.nu  # Number of controllable joints
        self.state_dim = (self.model.nbody - 1) * (3 + 4 + 3 + 3) + 1
        print("n", self.model.nu, self.model.nq, self.model.nv, self.model.nsite, self.model.nbody - 1)

        # self.ref_motion = self.load_reference_motion("backflip.npy")  # Load MoCap reference motion
        self.ref_motion = motion

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("init", self.dt, self.policy_freq, self.action_dim, self.state_dim)

    def load_reference_motion(self, path):
        """Load reference motion for backflip."""
        return np.load(path)  # Array of joint angles over time
    
    def step(self, action):
        """Apply action and step the simulation."""
        self.apply_pd_control(action)

        for _ in range(int(self.policy_freq / self.dt)):  # Simulate multiple steps per policy step
            mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.check_termination()
        return obs, reward, done

    def reset(self):
        """Reset simulation to a random frame from the reference motion."""
        mujoco.mj_resetData(self.model, self.data)
        frame_idx = np.random.randint(0, len(self.ref_motion))  # Reference State Initialization (RSI)

        frame_idx = 0

        self.data.qpos[:] = self.ref_motion["qpos"][frame_idx]
        # self.data.qpos[:] = np.random.rand(self.model.nq)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self):
        """Return current joint angles and velocities."""

        progress = torch.tensor([(self.data.time % self.ref_motion["dur"]) / self.ref_motion["dur"]]).to(self.device)
        pos = torch.tensor(self.data.xpos[1:]).to(self.device).flatten()
        quat = torch.tensor(self.data.xquat[1:]).to(self.device).flatten()
        lvel = torch.tensor(self.data.cvel[1:, :3]).to(self.device).flatten()
        rvel = torch.tensor(self.data.cvel[1:, 3:]).to(self.device).flatten()

        obs = torch.cat([progress, pos, quat, lvel, rvel])
        return obs
        return np.concatenate([self.data.qpos, self.data.qvel])

    def apply_pd_control(self, action):
        """Apply PD control to each joint."""
        # kp, kd = 100, 10  # PD gains
        # for i in range(self.action_dim):
        #     torque = kp * (action[i] - self.data.qpos[i]) - kd * self.data.qvel[i]
        #     self.data.ctrl[i] = torque  # Apply torque

        self.data.ctrl[:] = action.tolist()

    def compute_reward(self):
        """Compute imitation reward by comparing current pose to reference motion."""
        ref_idx = min(len(self.ref_motion) - 1, int(self.data.time / self.policy_freq))  # Match reference frame
        ref_quats = self.ref_motion["rots"][ref_idx]

        res = [
            np.linalg.norm(quaternion_difference(self.data.xquat[i], ref_quats[i])) ** 2 
            for i in range(len(ref_quats))
        ]
        
        pose_diff = np.sum(res)  # Pose matching
        reward = np.exp(-2 * pose_diff)

        return reward

    def check_termination(self):
        """Terminate episode if character falls."""
        if self.data.time >= self.ref_motion["horizon"]:
            return True
        return self.data.qpos[2] < -0.6  # Root height too low → character fell

    def render(self):
        """Use MuJoCo's new rendering API."""
        self.renderer.update_scene(self.data)
        return self.renderer.render()