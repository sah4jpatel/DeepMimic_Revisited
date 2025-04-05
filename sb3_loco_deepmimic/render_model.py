#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import gymnasium as gym
import cv2
import torch
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import BaseHumanoid from your LocoMujoco environment
from loco_mujoco.environments.humanoids.base_humanoid import BaseHumanoid

# -----------------------------
# Relaxed Fall Detection (required for environment)
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
# Environment Factory for Rendering
# -----------------------------
def make_env():
    def _init():
        env = gym.make("LocoMujoco",
                       env_name="HumanoidTorque.run",
                       dataset_type="perfect",
                       render_mode="rgb_array")
        return env
    return _init

def main(model_path, output_video, num_steps=1000):
    # Create a vectorized environment with one instance.
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_reward=True, norm_obs=False)
    
    # Load the model with the provided environment.
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    frames = []
    
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        # Render the frame.
        frame = env.render(mode="rgb_array")
        if frame is None:
            frame = env.unwrapped.render()
        # Rotate the frame 90° clockwise.
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Display the frame.
        cv2.imshow("Render", frame)
        cv2.waitKey(1)
        frames.append(frame)
        # Check termination.
        term_flag = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated
        trunc_flag = truncated[0] if isinstance(truncated, (list, np.ndarray)) else truncated
        if term_flag or trunc_flag:
            break

    env.close()
    cv2.destroyAllWindows()

    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Saved render video at {output_video}")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render and save video of DeepMimic model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (zip).")
    parser.add_argument("--output", type=str, default="rendered_run.mp4", help="Output video filename.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run for rendering.")
    args = parser.parse_args()
    main(args.model, args.output, args.steps)
