#%%


import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

#%%

import sys
sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/src/training')
sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/mujoco')

from PPO_Agent import PPOAgent
from humanoid import HumanoidEnv

#%%

def train_ppo_on_humanoid(model_path, reference_data_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)


    env = HumanoidEnv(
        model_path=model_path,
        reference_data_path=reference_data_path
    )

    obs_dim = env._get_obs().shape[0]
    act_dim = env.data.ctrl.shape[0]


    
    max_episodes = 1000
    steps_per_rollout = 2**11
    agent = PPOAgent(obs_dim, act_dim, lr=1e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, train_iters=10)
    
    
    reward_history = []
    
    
    plot_interval = 50
    checkpoint_interval = 50
    
    for episode in range(max_episodes):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        
        obs = env.reset()
        ep_rew = 0
        for step in range(steps_per_rollout):
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(done)
            
            obs = next_obs
            ep_rew += reward
            
            if done:
                obs = env.reset()
        
        _, _, last_val = agent.get_action(obs)
        
        advantages = agent.compute_advantages(rew_buf, val_buf, done_buf, last_val)
        returns = [v + adv for v, adv in zip(val_buf, advantages)]
        
        agent.update(obs_buf, act_buf, logp_buf, returns, advantages)

        
        
        
        print(f"Episode {episode}, total reward: {ep_rew:.2f}")
        reward_history.append(ep_rew)
        
        if episode % plot_interval == 0 and episode > 0:
            plt.figure()
            plt.plot(reward_history, label='Episode Reward')
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Training Progress")
            plt.legend()
            plt.grid(True)
            plot_filename = f"training_progress_{episode}.png"
            plt.savefig(os.path.join(save_dir, plot_filename))
            plt.close()
            print(f"Saved training progress graph: {plot_filename}")

            actions_df = pd.DataFrame(act_buf)
            csv_path = os.path.join(save_dir, f"actions_episode_{episode}.csv")
            actions_df.to_csv(csv_path, index=False)
            print(f"Saved actions to CSV: {csv_path}")
        
        
        if episode % checkpoint_interval == 0 and episode > 0:
            checkpoint_path = f"checkpoint_{episode}.pth"
            agent.save(os.path.join(save_dir,checkpoint_path)) 
            print(f"Saved checkpoint to {checkpoint_path}")

#%%

if __name__ == "__main__":
    model_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/envs/dp_env_v2.xml'
    reference_data_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/motions/humanoid3d_walk.txt'
    save_dir = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/exp/exp14'
    train_ppo_on_humanoid(model_path,reference_data_path,save_dir)

#%%