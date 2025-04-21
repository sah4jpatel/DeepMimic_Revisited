from DRL import DRL
from PIL import Image
import os
from customEnv import MyEnv
import mujoco
import tqdm
import torch
from PPOAgent import PPOAgent
from mocap.mocap import MocapDM
import numpy as np
from copy import deepcopy

os.makedirs("videos_new10",exist_ok=True)

MOCAP = MocapDM()
MOCAP.load_mocap("walk_long.txt")
# print(len(MOCAP.data_config))
ref_states = np.hstack((np.array(MOCAP.data_config),np.array(MOCAP.data_vel)))
# print(ref_states.shape)
states_mean = np.mean(ref_states,axis = 0)
# print(states_mean.shape)
states_std = np.std(ref_states,axis=0)
# print(states_std.shape)

def save_render_image(image,ind):
    outpath = os.path.join("./videos_new10",f"{ind}.png")
    im = Image.fromarray(image)
    im.save(outpath)
agent = PPOAgent(35+34+160+1,28,.95)
# checkpoint = torch.load("/Users/xiaowenma/GT/Classes/CS 8803 DRL/project/deepmimic_pytorch/test_mujoco/mujoco copy/saved_models/1743986221.118161/model_checkpoint_60.pth")

# checkpoint = torch.load("/Users/xiaowenma/GT/Classes/CS 8803 DRL/project/deepmimic_pytorch/test_mujoco/mujoco copy/saved_models/1744048850.1792262/model_checkpoint_40.pth")

checkpoint = torch.load("/Users/xiaowenma/GT/Classes/CS 8803 DRL/project/deepmimic_pytorch/test_mujoco/mujoco copy/saved_models/1744823345.483511/model_checkpoint_1000.pth")


agent.value.load_state_dict(checkpoint['value_state_dict'])
agent.policy.load_state_dict(checkpoint['policy_state_dict'])
agent.value_out.load_state_dict(checkpoint['value_out_state_dict'])
agent.policy_out.load_state_dict(checkpoint['policy_out_state_dict'])
obs_mean = checkpoint['obs_mean']
obs_std = torch.sqrt(checkpoint['obs_var'])

# value_optimizer.load_state_dict(checkpoint['optimizer_value_state_dict'])
# policy_optimizer.load_state_dict(checkpoint['optimizer_policy_state_dict'])

env = MyEnv("")
# Create environment with proper render_mode

obs, _ = env.reset()
obs = torch.Tensor(obs)

with mujoco.Renderer(env.model) as renderer:

  for t in tqdm.tqdm(range(90)):

    # obs_normalized = deepcopy(obs)
    # obs_normalized[:69] = (obs_normalized[:69]-states_mean)/(states_std+1e-8)

    with torch.no_grad():
       actions, _ = agent.get_action(obs)
      # actions, _ = agent.get_action(obs_normalized)  # Get action from policy

    next_obs, rewards, done, truncated, infos = env.step(actions.numpy())

    if done:
        # self.writer.add_scalar("Duration", t, i)
        break
    mujoco.mj_forward(env.model,env.data)
    renderer.update_scene(env.data)
    pixels = renderer.render()
    save_render_image(pixels,t)
    obs = torch.Tensor(next_obs)


env.close()