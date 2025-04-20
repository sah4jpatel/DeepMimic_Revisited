#!/usr/bin/env python3

import argparse
import time

import torch
import torch.nn as nn
import mujoco
import mujoco.viewer

from env import MuJoCoBackflipEnv
from main2 import GATPolicyNetwork, load_data
from utils import motion_to_posvel
from ppo import PPO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize trained GAT‐PPO checkpoint in MuJoCo env"
    )
    parser.add_argument(
        "--xml", type=str, default="data/humanoid.xml",
        help="Path to MuJoCo model XML file"
    )
    parser.add_argument(
        "--motion", type=str, default="data/walk.txt",
        help="Path to reference motion JSON file"
    )
    parser.add_argument(
        "--actor_ckpt", type=str, default="ckpt/ppo_actor.pth",
        help="Path to actor checkpoint"
    )
    parser.add_argument(
        "--critic_ckpt", type=str, default="ckpt/ppo_critic.pth",
        help="Path to critic checkpoint"
    )
    parser.add_argument(
        "--seed", type=int, default=14,
        help="Random seed"
    )
    parser.add_argument(
        "--reset_frame", type=int, default=0,
        help="Frame index to reset to when visualizing"
    )
    parser.add_argument(
        "--hz", type=float, default=30.0,
        help="Control frequency in Hz"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)


    motion = load_data(args.motion)
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(mj_model)
    ref = motion_to_posvel(motion["Frames"], mj_model, mj_data)



    env = MuJoCoBackflipEnv(args.xml, ref)



    policy = GATPolicyNetwork(env, in_dim=4, hidden_dim=64, action_dim=env.action_dim)
    critic = nn.Sequential(
        nn.Linear(env.state_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )


    policy.load_state_dict(torch.load(args.actor_ckpt, map_location="cpu"))
    critic.load_state_dict(torch.load(args.critic_ckpt, map_location="cpu"))


    ppo = PPO(policy, critic, env)


    state = env.reset(frame_idx=args.reset_frame)
    dt = 1.0 / args.hz


    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            with torch.no_grad():
                action, _ = ppo.get_action(state, sample=False)

            good, next_state, reward, done = env.step(action)
            if not good:
                state = env.reset(frame_idx=args.reset_frame)
                continue

            viewer.sync()
            time.sleep(dt)

            state = next_state
            if done:
                state = env.reset(frame_idx=args.reset_frame)

if __name__ == "__main__":
    main()
