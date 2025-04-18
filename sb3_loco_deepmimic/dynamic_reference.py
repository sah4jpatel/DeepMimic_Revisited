#!/usr/bin/env python3
import os
import sys
import time
import math
import uuid
import argparse
import numpy as np
import gymnasium as gym
import torch
import cv2
import mujoco  # MuJoCo Python bindings

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter

# Import the BaseHumanoid from your LocoMujoco package.
from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid

# -----------------------------
# Directories for models, logs, videos
# -----------------------------
video_folder = "videos"
models_folder = "models"
logs_folder = "logs"
for folder in [video_folder, models_folder, logs_folder]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# Relaxed Fall Detection Override
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

# Override the original _has_fallen method.
BaseHumanoid._has_fallen = relaxed_has_fallen

# -----------------------------
# Flat Terrain Wrapper
# -----------------------------
class FlatTerrainWrapper(gym.Wrapper):
    """
    Overrides the terrain in the loaded model so that the heightfield is made flat.
    This is done by setting all heightfield_data to zero.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.make_flat()
        return obs, info

    def make_flat(self):
        env_unwrapped = self.env.unwrapped
        model = None
        if hasattr(env_unwrapped, "sim"):
            try:
                model = env_unwrapped.sim.model
            except Exception:
                pass
        if model is None and hasattr(env_unwrapped, "model"):
            model = env_unwrapped.model
        if model is None and hasattr(env_unwrapped, "_model"):
            model = env_unwrapped._model
        if model is None:
            raise AttributeError("Cannot access the MuJoCo model in FlatTerrainWrapper.")
        
        # Find the heightfield asset "terrain" using mj_name2id;
        # if found, override its data to zeros.
        hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hfield_id != -1:
            flat_values = np.zeros_like(model.hfield_data, dtype=np.float64)
            model.hfield_data[:] = flat_values
        else:
            print("FlatTerrainWrapper: HFIELD 'terrain' not found.")

# -----------------------------
# Additional Wrappers for rendering, scaling, and reward shaping
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
# Function to create a flat terrain environment with wrappers
# -----------------------------
def make_flat_env():
    def _init():
        env = gym.make("LocoMujoco",
                       env_name="HumanoidTorque.run",  # your custom MJCF that includes heightfield "terrain"
                       dataset_type="perfect",
                       render_mode="rgb_array")
        # Force flat terrain.
        env = FlatTerrainWrapper(env)
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

def main():
    parser = argparse.ArgumentParser(
        description="Run model validation in a flat-terrain environment and record a video clip."
    )
    parser.add_argument("model_path", type=str, help="Path to the trained model checkpoint (.zip) file.")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration of the video in seconds (default: 10)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for video recording (default: 30)")
    args = parser.parse_args()

    # Set up the evaluation environment.
    env = DummyVecEnv([make_flat_env()])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)
    
    # Load the trained model using the provided path.
    model = PPO.load(args.model_path, env=env, device="cpu")
    
    # Video recording parameters.
    fps = args.fps
    duration = args.duration  # in seconds
    num_frames = fps * duration
    frames = []
    
    obs = env.reset()
    start_time = time.time()
    
    for i in range(num_frames):
        # Predict the next action using the loaded model.
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment.
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        
        # Render the frame (RGB array).
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        
        if terminated or truncated:
            obs = env.reset()
    
    end_time = time.time()
    print(f"Recorded {len(frames)} frames in {end_time - start_time:.2f} seconds.")
    
    # Save the recorded frames to a video file using OpenCV.
    # Note: OpenCV's VideoWriter expects frames in BGR, so we convert each frame.
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = os.path.join(video_folder, "validation_run.mp4")
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert the frame from RGB to BGR before writing.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    video_writer.release()
    
    print(f"Saved validation video to {video_filename}.")

if __name__ == "__main__":
    main()
