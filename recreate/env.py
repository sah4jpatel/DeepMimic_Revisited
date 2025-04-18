import gymnasium as gym
import mujoco
# import mujoco_py
import numpy as np
import torch
from utils import quaternion_difference
import math
import random

class MuJoCoBackflipEnv(gym.Env):
    def __init__(self, xml, motion):
        super(MuJoCoBackflipEnv, self).__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.integrator = 1

        self.model2 = mujoco.MjModel.from_xml_path(xml)
        self.data2 = mujoco.MjData(self.model2)
        self.model2.opt.timestep = 0.000001

        self.model.opt.timestep = 0.005
        self.dt = self.model.opt.timestep  # MuJoCo timestep
        self.policy_freq = 1/30  # 30 Hz control frequency
        self.action_dim = self.model.nu  # Number of controllable joints
        self.state_dim = (self.model.nbody - 1) * (3) + 1 + self.model.nq + self.model.nv

        self.ref_motion = motion

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 0 -   2:   root positions
        # 3 -   6:   root rotation
        # 7 -   9:   chest
        # 10 -  12:  neck
        # 13 -  15:  right shoulder
        #       16:  right elbow
        # 17 -  19:  left shoulder
        #       20:  left elbow
        # 21 -  23:  right hip
        #       24:  right knee
        # 25 -  27:  right ankle
        # 28 -  30:  left hip
        #       31:  left knee
        # 32 -  34:  left ankle 
        self.torque_limits = torch.tensor([
            200, 200, 200,
            50, 50, 50,
            100, 100, 100,
            60,
            100, 100, 100,
            60,
            200, 200, 200,
            150,
            90, 90, 90,
            200, 200, 200,
            150,
            90, 90, 90
        ])

        self.kp = torch.tensor([
            1000, 1000, 1000,
            100, 100, 100,
            400, 400, 400,
            300,
            400, 400, 400,
            300,
            500, 500, 500,
            500,
            400, 400, 400,
            500, 500, 500,
            500,
            400, 400, 400,
        ])
        self.kd = self.kp / 10

        self.penalty_weight = 0
        
    def step(self, action):
        """Apply action and step the simulation."""
        self.apply_pd_control(action)

        lt, fail = self.data.time, False
        for _ in range(int(self.policy_freq / self.dt)):  # Simulate multiple steps per policy step
        # for _ in range(1):  # Simulate multiple steps per policy step
            mujoco.mj_step(self.model, self.data)
            if self.data.time < lt:
                fail = True
                break
            lt = self.data.time

        if fail:
            return False, 0, 0, True

        obs = self.get_observation()
        reward = self.compute_reward(action)
        done = self.check_termination()
        return True, obs, reward, done

    def reset(self, frame_idx = -1):
        """Reset simulation to a random frame from the reference motion."""
        mujoco.mj_resetData(self.model, self.data)
        if frame_idx == -1:
            frame_idx = np.random.randint(0, len(self.ref_motion))  # Reference State Initialization (RSI)

        self.data.time = frame_idx * self.ref_motion["freq"]

        self.data.qpos[:] = self.ref_motion["qpos"][frame_idx]
        self.data.qvel[:] = self.ref_motion["angvels"][frame_idx]
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self):
        """Return current joint angles and velocities."""

        root_pos = self.data.xpos[1]  # Root world position
        root_quat = self.data.xquat[1]  # Root world orientation (quaternion)
        
        # Convert root orientation to rotation matrix (to define local frame)
        root_mat = np.zeros((9))
        mujoco.mju_quat2Mat(root_mat, root_quat)  # Convert quat to rotation matrix
        root_mat = root_mat.reshape((3, 3))
        
        poss, quats, lvels = [], [], []

        poss.extend(root_pos.tolist())

        for i in range(2, self.model.nbody):  # Skip world body (0)
            # **1. Relative Position**
            pos_rel = self.data.xpos[i] - root_pos  # World position - root position
            pos_rel = root_mat.T @ pos_rel  # Transform to local frame

            # **2. Rotation (Quaternion)**
            # quat = self.data.xquat[i]  # Direct quaternion

            # **3. Linear and Angular Velocities (Local Frame)**
            # lin_vel = self.data.cvel[i, 3:] - root_vel  # Velocity relative to root
            # lin_vel = root_mat.T @ lin_vel  # Transform to local frame
            # lin_vel = np.sign(lin_vel) * np.log(1 + np.abs(lin_vel))

            # ang_vel = self.data.cvel[i, :3] - root_ang_vel  # Angular velocity relative to root
            # ang_vel = root_mat.T @ ang_vel  # Transform to local frame
            # ang_vel = np.sign(ang_vel) * np.log(1 + np.abs(ang_vel))

            # Store features
            # observations.extend(pos_rel.tolist())
            # observations.extend(quat.tolist())
            # observations.extend(lin_vel.tolist())
            # observations.extend(ang_vel.tolist())

            poss.extend(pos_rel.tolist())
            # quats.extend(quat.tolist())
            # lvels.extend(lin_vel.tolist())

        # observations = [(self.data.time % self.ref_motion["dur"]) / self.ref_motion["dur"], *poss, *quats, *lvels]
        # return torch.tensor(observations, dtype=torch.float).to(self.device)

        poss = torch.tensor(poss, dtype=torch.float).to(self.device).flatten()

        progress = torch.tensor([(self.data.time % self.ref_motion["dur"]) / self.ref_motion["dur"]]).to(self.device)
        # pos = torch.tensor(self.data.xpos[1:]).to(self.device).flatten()
        # quat = torch.tensor(self.data.xquat[1:]).to(self.device).flatten()
        qpos = torch.tensor(self.data.qpos).to(self.device).flatten()
        # rvel = torch.tensor(self.data.cvel[1:, :3]).to(self.device).flatten()
        qvel = torch.tensor(self.data.qvel).to(self.device).flatten()
        # lvel = torch.tensor(self.data.cvel[1:, 3:]).to(self.device).flatten()

        # obs = torch.cat([progress, pos, quat, lvel, rvel]).to(torch.float)
        obs = torch.cat([progress, poss, qpos, qvel]).to(torch.float)
        return obs
        # return np.concatenate([self.data.qpos, self.data.qvel])

    def apply_pd_control(self, action):
        """Apply PD control to each joint."""
        # ref_idx = math.floor((self.data.time % self.ref_motion["dur"]) / self.ref_motion["dur"] * len(self.ref_motion["qpos"]))
        # ref_qp = np.array(self.ref_motion["qpos"][ref_idx][7:])

        qp = self.data.qpos[7:]
        qv = self.data.qvel[6:]
        # torque = self.kp * (ref_qp + action.tolist() - qp) * 0.3 - self.kd * qv * 0.02
        torque = self.kp * (action.tolist() - qp) - self.kd * qv
        # torque = self.kp * action.cpu() * 0.25 - self.kd * qv * 0.1
        self.data.ctrl[:] = (torque / self.torque_limits).clip(-1, 1).tolist()
        # self.data.ctrl[:] = action.tolist()

    def compute_reward(self, action):
        """Compute imitation reward by comparing current pose to reference motion."""
        ref_idx = math.floor((self.data.time % self.ref_motion["dur"]) / self.ref_motion["dur"] * len(self.ref_motion["qpos"]))
        ref_quats = self.ref_motion["rots"][ref_idx]
        quats = self.data.xquat[1:]

        ref_qp = self.ref_motion["qpos"][ref_idx]
        qp = self.data.qpos
        r_qp = (2 * np.arccos(np.clip(quaternion_difference(qp[3:7], ref_qp[3:7])[3], -1, 1)))

        qp_diff = (qp[7:] - ref_qp[7:])
        qp_diff = np.concatenate([qp_diff, r_qp[None]])
        qp_rew = np.exp(-2 * 0.5 * np.sum(qp_diff ** 2))
        
        ref_com = self.ref_motion["qpos"][ref_idx][2]
        com = self.data.qpos[2]

        com_diff = com - ref_com
        com_rew = np.exp(-10 * np.linalg.norm(com_diff) ** 2)

        # ref_lvel = self.ref_motion["linvels"][ref_idx]
        # lvel = self.data.cvel[1:, :3]

        # lvel_diff = np.linalg.norm(lvel - ref_lvel, axis=-1)
        # lvel_rew = np.exp(-2 * np.mean(lvel_diff ** 2))

        ref_ends = self.ref_motion["ends"][ref_idx]
        ends = self.data.geom_xpos[[6, 9, 12, 15]]
        
        ends_diff = np.linalg.norm(ends - ref_ends, axis=-1)
        ends_rew = np.exp(-40 * np.sum(ends_diff ** 2))

        ref_avel = self.ref_motion["angvels"][ref_idx][3:]
        avel = self.data.qvel[3:]

        avel_diff = avel - ref_avel
        avel_rew = np.exp(-0.1 * 0.5 * np.sum(avel_diff ** 2))

        action_pen = torch.mean(action ** 2).item()

        reward = (0.65 * qp_rew + 0.1 * com_rew + 0.1 * avel_rew + 0.15 * ends_rew) / 1 - self.penalty_weight * action_pen
        reward = reward.clip(0, 1)

        return reward

    def check_termination(self):
        """Terminate episode if character falls."""
        if self.data.time >= self.ref_motion["horizon"]:
            return True
        # return False
        return (self.data.geom_xpos[[1, 2, 3], 2] < -0.75).any()