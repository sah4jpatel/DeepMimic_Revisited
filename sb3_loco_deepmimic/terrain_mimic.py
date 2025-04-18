#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import gymnasium as gym
import torch
import cv2
import uuid
import mujoco  # ensure you have the correct MuJoCo Python bindings

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter

# Import BaseHumanoid from your LocoMujoco environment
from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid

# -----------------------------
# Directories
# -----------------------------
video_folder = "videos"
models_folder = "models"
logs_folder = "logs"
for folder in [video_folder, models_folder, logs_folder]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# Relaxed Fall Detection
# -----------------------------
def relaxed_has_fallen(self, obs, return_err_msg=False):
    pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])
    pelvis_height_cond = (obs[0] < -0.6) or (obs[0] > 0.2)
    pelvis_tilt_cond = (pelvis_euler[0] < -np.pi/3) or (pelvis_euler[0] > np.pi/6)
    pelvis_list_cond = (pelvis_euler[1] < -np.pi/6) or (pelvis_euler[1] > np.pi/4)
    pelvis_rot_cond = (pelvis_euler[2] < -np.pi/4) or (pelvis_euler[2] > np.pi/4)
    pelvis_condition = pelvis_height_cond or pelvis_tilt_cond or pelvis_list_cond or pelvis_rot_cond
    lumbar_euler = self._get_from_obs(obs, ["q_lumbar_extension", "q_lumbar_bending", "q_lumbar_rotation"])
    lumbar_ext_cond = (lumbar_euler[0] < -np.pi/3) or (lumbar_euler[0] > np.pi/8)
    lumbar_bend_cond = (lumbar_euler[1] < -np.pi/8) or (lumbar_euler[1] > np.pi/8)
    lumbar_rot_cond = (lumbar_euler[2] < -np.pi/3) or (lumbar_euler[2] > np.pi/3)
    lumbar_condition = lumbar_ext_cond or lumbar_bend_cond or lumbar_rot_cond
    return pelvis_condition or lumbar_condition

BaseHumanoid._has_fallen = relaxed_has_fallen

# -----------------------------
# Dynamic Terrain Wrapper
# -----------------------------
class DynamicTerrainWrapper(gym.Wrapper):
    """
    This Gym wrapper adds dynamic terrain variations into a MuJoCo environment.
    It accesses the underlying MuJoCo model via one of several attributes ('sim.model', 'model', or '_model').
    Then, it:
      - Randomizes the "floor" geom's inclination.
      - Modifies the heightfield data of a geom (asset) named "terrain" by combining noise with a sinusoidal pattern.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.randomize_terrain()
        return obs, info

    def randomize_terrain(self):
        env_unwrapped = self.env.unwrapped
        model = None
        if hasattr(env_unwrapped, 'sim'):
            try:
                model = env_unwrapped.sim.model
            except Exception:
                pass
        if model is None and hasattr(env_unwrapped, 'model'):
            model = env_unwrapped.model
        if model is None and hasattr(env_unwrapped, '_model'):
            model = env_unwrapped._model
        if model is None:
            raise AttributeError("Cannot access the MuJoCo model. "
                                 "Check if the environment exposes it via 'sim.model', 'model', or '_model'.")

        # --- Example 1: Modify ground incline (geom "floor") ---
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id != -1:
            # Random pitch angle between -15 and 15 degrees.
            pitch_deg = np.random.uniform(-15, 15)
            pitch_rad = np.radians(pitch_deg)
            # Create quaternion for rotation about y-axis [w,x,y,z]
            quat = np.array([np.cos(pitch_rad/2), 0, np.sin(pitch_rad/2), 0], dtype=np.float64)
            model.geom_quat[floor_id] = quat
        else:
            print("Floor geom 'floor' not found.")

        # --- Example 2: Modify heightfield (asset and geom "terrain") ---
        hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hfield_id == -1:
            # For debugging, list available heightfield names
            hfield_names = []
            try:
                for i in range(model.nhfield):
                    hfield_names.append(model.asset.hfield_names[i])
            except Exception as e:
                hfield_names = str(e)
            print("HFIELD NOT FOUND. Available heightfield names:", hfield_names)
        else:
            try:
                asset_id = mujoco.mj_name2id(model.asset, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
                nrow = int(model.asset.hfield_nrow[asset_id])
                ncol = int(model.asset.hfield_ncol[asset_id])
            except Exception:
                nrow, ncol = 64, 64  # Use default values if asset info is unavailable

            noise = np.random.uniform(-0.05, 0.05, size=(nrow * ncol))
            x = np.linspace(0, 2*np.pi, ncol)
            ridge_pattern = 0.05 * np.sin(3 * x)
            ridge_pattern = np.tile(ridge_pattern, nrow)
            new_hfield = noise + ridge_pattern
            model.hfield_data[:] = new_hfield.astype(np.float64)

# -----------------------------
# Additional Wrappers
# -----------------------------
class RotatedRenderWrapper(gym.Wrapper):
    def render(self):
        frame = self.env.render()
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

class ScaledActionEnv(gym.Wrapper):
    def __init__(self, env, scale=0.6):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        scaled_action = np.clip(action * self.scale, -1.0, 1.0)
        return self.env.step(scaled_action)

class DeepMimicRewardWrapper(gym.Wrapper):
    def __init__(self, env, reference_data, state_mean, state_std, action_mean, action_std,
                 w_p=2.5, w_v=0.05, w_e=0.5, w_tc=0.05, fall_penalty=-10.0):
        super().__init__(env)
        self.reference_data = reference_data
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.w_p = w_p
        self.w_v = w_v
        self.w_e = w_e
        self.w_tc = w_tc
        self.fall_penalty = fall_penalty
        self.current_idx = 0
        self.max_idx = len(reference_data["states"]) - 1
        self.state_dim = reference_data["states"].shape[1]
        self.half_dim = self.state_dim // 2
        self.pos_indices = list(range(0, self.half_dim))
        self.vel_indices = list(range(self.half_dim, self.state_dim))
        self.episode_rewards = []
        self.episode_lengths = []
        self.running_reward = 0
        self.episode_len = 0
        self.global_step = 0
        self.writer = SummaryWriter(os.path.join(logs_folder, f"env_{uuid.uuid4().hex[:6]}"))
        self.prev_obs = None

    def reset(self, **kwargs):
        if hasattr(self.env.unwrapped, "random_start"):
            self.env.unwrapped.random_start = True
        obs, info = self.env.reset(**kwargs)
        if self.episode_len > 0:
            self.episode_rewards.append(self.running_reward)
            self.episode_lengths.append(self.episode_len)
            if len(self.episode_rewards) >= 10:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                self.writer.add_scalar("rollout/ep_rew_mean", mean_reward, self.global_step)
                self.writer.add_scalar("rollout/ep_len_mean", mean_length, self.global_step)
        self.running_reward = 0
        self.episode_len = 0
        self.prev_obs = None
        self.current_idx = np.random.randint(0, self.max_idx + 1)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ref_state = self.reference_data["states"][min(self.current_idx, self.max_idx)]
        ref_action = self.reference_data["actions"][min(self.current_idx, self.max_idx)]
        norm_obs = (obs - self.state_mean) / self.state_std
        norm_ref = (ref_state - self.state_mean) / self.state_std
        norm_action = (action - self.action_mean) / self.action_std
        norm_ref_action = (ref_action - self.action_mean) / self.action_std
        pose_error = np.mean(np.square(norm_obs[self.pos_indices] - norm_ref[self.pos_indices]))
        vel_error = np.mean(np.square(norm_obs[self.vel_indices] - norm_ref[self.vel_indices]))
        foot_indices = [7, 8, 12, 13]
        ee_error = np.mean(np.square(norm_obs[foot_indices] - norm_ref[foot_indices]))
        action_error = np.clip(np.mean(np.square(norm_action - norm_ref_action)), 0, 10.0)
        pose_reward = math.exp(-0.5 * pose_error)
        vel_reward = math.exp(-0.1 * vel_error)
        ee_reward = math.exp(-0.5 * ee_error)
        action_reward = math.exp(-0.1 * action_error)
        continuity_bonus = 0.05
        phase_progress_reward = self.current_idx / self.max_idx
        imitation_reward = (self.w_p * pose_reward +
                            self.w_v * vel_reward +
                            self.w_e * ee_reward +
                            0.05 * action_reward)
        task_reward = self.w_tc * reward
        combined_reward = imitation_reward + task_reward + continuity_bonus + 0.2 * phase_progress_reward

        if self.prev_obs is not None:
            smoothness_penalty = -0.01 * np.linalg.norm(obs - self.prev_obs)
            combined_reward += smoothness_penalty
            info["smoothness_penalty"] = smoothness_penalty
        self.prev_obs = obs.copy()

        if self.env.unwrapped._has_fallen(obs):
            combined_reward += self.fall_penalty
            info["fall_penalty"] = self.fall_penalty
            terminated = True

        self.current_idx += 1
        self.running_reward += combined_reward
        self.episode_len += 1
        self.global_step += 1
        return obs, combined_reward, terminated, truncated, info

    def __del__(self):
        self.writer.close()

# -----------------------------
# Environment Factory Function
# -----------------------------
def make_env():
    def _init():
        env = gym.make("LocoMujoco",
                       env_name="HumanoidTorque.run",
                       dataset_type="perfect",
                       render_mode="rgb_array")
        # Apply dynamic terrain variations.
        env = DynamicTerrainWrapper(env)
        dataset = env.unwrapped.create_dataset()
        state_mean = np.mean(dataset["states"], axis=0)
        state_std = np.std(dataset["states"], axis=0) + 1e-8
        action_mean = np.mean(dataset["actions"], axis=0)
        action_std = np.std(dataset["actions"], axis=0) + 1e-8

        env = DeepMimicRewardWrapper(env, dataset, state_mean, state_std, action_mean, action_std)
        env = RotatedRenderWrapper(env)
        env = ScaledActionEnv(env, scale=0.6)
        return env
    return _init

# -----------------------------
# Custom Render Callback to Record Video
# -----------------------------
class RenderAndRecordCallback(BaseCallback):
    def __init__(self, render_env_fn, record_freq, video_length=300, video_folder=video_folder, verbose=0):
        super().__init__(verbose)
        self.render_env_fn = render_env_fn
        self.record_freq = record_freq
        self.video_length = video_length
        self.video_folder = video_folder

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_freq < self.locals.get("n_steps", 1):
            self._record_video()
        return True

    def _record_video(self):
        env = self.render_env_fn()
        obs, info = env.reset()
        frames = []
        for step in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)
            if terminated or truncated:
                break
        env.close()
        if frames:
            height, width, _ = frames[0].shape
            video_filename = os.path.join(self.video_folder, f"render_{self.num_timesteps}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (width, height))
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            if self.verbose > 0:
                print(f"Saved render video at {video_filename}")

# -----------------------------
# Parallel Environment and PPO Setup
# -----------------------------
num_envs = 10

checkpoint_callback = CheckpointCallback(
    save_freq=500_000,
    save_path=models_folder,
    name_prefix="humanoid_ppo_checkpoint"
)

total_timesteps = 1_000_000_000  # Adjust as needed

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    raw_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    vec_env = VecNormalize(raw_env, norm_reward=True, norm_obs=False)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=logs_folder,
        learning_rate=1e-4,
        n_steps=10000,
        batch_size=1000,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",  # Using CPU as specified
        target_kl=0.03
    )

    def make_render_env():
        env = gym.make("LocoMujoco",
                       env_name="HumanoidTorque.run",
                       dataset_type="perfect",
                       render_mode="rgb_array")
        env = DynamicTerrainWrapper(env)
        dataset = env.unwrapped.create_dataset()
        state_mean = np.mean(dataset["states"], axis=0)
        state_std = np.std(dataset["states"], axis=0) + 1e-8
        action_mean = np.mean(dataset["actions"], axis=0)
        action_std = np.std(dataset["actions"], axis=0) + 1e-8
        env = DeepMimicRewardWrapper(env, dataset, state_mean, state_std, action_mean, action_std)
        env = RotatedRenderWrapper(env)
        env = ScaledActionEnv(env, scale=0.6)
        return env

    render_callback = RenderAndRecordCallback(render_env_fn=make_render_env,
                                              record_freq=2_500_000,
                                              video_length=300,
                                              verbose=1)

    model.learn(total_timesteps=total_timesteps,
                callback=[checkpoint_callback, render_callback],
                tb_log_name="humanoid_deepmimic")
