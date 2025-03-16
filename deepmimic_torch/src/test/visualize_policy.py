#%%


from gym import spaces
import mujoco
from mujoco import MjData, MjModel
import mujoco_viewer

#%%

import sys
sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/src/training')
sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/mujoco')

from PPO_Agent import PPOAgent
from humanoid import HumanoidEnv



#%%
import time
import torch
import numpy as np
import mujoco
import mujoco_viewer  

def visualize_policy(checkpoint_path, model_xml, reference_data_path, num_episodes=3, episode_length=10000):


    env = HumanoidEnv(model_path=model_xml, reference_data_path=reference_data_path)
    obs_dim = env._get_obs().shape[0]
    act_dim = env.data.ctrl.shape[0]


    agent = PPOAgent(obs_dim, act_dim)
    agent.load(checkpoint_path)
    policy = agent.actor_critic
    policy.eval()

    model = env.model
    data = env.data


    viewer = mujoco_viewer.MujocoViewer(model, data)

    for ep in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0

        for t in range(episode_length):
            
            viewer.render()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor)
            action = action.numpy()
            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            obs = next_obs

            if done:
                break

        print(f"Episode {ep+1}, total reward: {episode_return:.2f}")


    viewer.close()


#%%
if __name__ == "__main__":
    visualize_policy(
        checkpoint_path="/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/exp/exp14/checkpoint_50.pth",
        model_xml= '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/envs/dp_env_v2.xml',
        reference_data_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/motions/humanoid3d_walk.txt',
        num_episodes=3,
        episode_length=1000
    )
# %%

