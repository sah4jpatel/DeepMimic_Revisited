#%%

import gym
import numpy as np
import os
from gym import spaces
import mujoco
from mujoco import MjData, MjModel
import mujoco_viewer

#%%

from data_reader import DataReader

#%%




class HumanoidEnv(gym.Env):


    def __init__(self,
                 model_path=None,
                 reference_data_path=None,
                 frame_skip=5,
                 reward_scale=1.0):
        super(HumanoidEnv, self).__init__()

        self.mocap = DataReader()

        if reference_data_path is not None:
            self.mocap.load_mocap(reference_data_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.frame_skip = frame_skip
        self.reward_scale = reward_scale
        self.reference_index = 0
        self.viewer = None

    def _get_obs(self):
        qpos = self.data.qpos.copy()[:2].astype(np.float32) #EDIT leave root
        qvel = self.data.qvel.copy().astype(np.float32) #EDIT leave root
        # ref = self.mocap.data[self.reference_index]
        return np.concatenate([qpos, qvel]) #, ref])

    def step(self, action):

        frame_duration = self.mocap.data[self.reference_index, 0]
        sim_dt = self.model.opt.timestep
        num_sim_steps = int(self.mocap.dt//sim_dt)

        # print(num_sim_steps)
        # print(self.mocap.data[self.reference_index, 0])
        # print(self.mocap.dt)
        # print(sim_dt)
        # print(self.mocap.dt//sim_dt)

        

        for it in range(num_sim_steps):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

        # action = self.prev_action

        # for _ in range(self.frame_skip): #EDIT for smooth motion

        # self.data.ctrl[:] = action
        # mujoco.mj_step(self.model, self.data)
        
 
 
        self.reference_index += 1
        done = False
        if self.reference_index >= len(self.mocap.data):
            done = True
            self.reference_index = 0 # EDIT add recycle

        if done == False:
            done = self.is_healthy()
            
        current_qpos = self.data.qpos.copy()
        ref_qpos = self.mocap.data_qpos[self.reference_index]
        pos_error  = np.linalg.norm(current_qpos[3:] - ref_qpos[3:])
        vel_error  = np.linalg.norm(self.data.qvel - self.mocap.data_vel[self.reference_index])
        reward = 2 * np.exp(-pos_error) + 0.2 * np.exp(-vel_error)

        if done == False:
            done = self.is_healthy()
            if done == True:
                mass = np.expand_dims(self.model.body_mass, 1)
                xpos = self.data.xipos
                z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
                reward += np.exp(-np.linalg.norm(0.8-z_com) * 100) 
                

        obs = self._get_obs()
        return obs, reward, done, {}

    def is_healthy(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.data.xipos
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
        stop = bool((z_com < 0.7) or (z_com > 2.0))
        if stop == True:
                print(f"ZCOM Value: {z_com}" )

        return stop
    


    def reset(self):
        self.reference_index = 0
        init_qpos = self.mocap.data_qpos[self.reference_index]
        self.data.qpos[:] = init_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -20
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


#%%
if __name__ == '__main__':

    import sys

    sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/external_code')
    from transformations import quaternion_from_euler, euler_from_quaternion

    env_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/envs/dp_env_v2.xml'
    # env_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/humanoid/humanoid.xml'
    motion_file_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/motions/humanoid3d_walk.txt'
    env = HumanoidEnv(env_file_path,motion_file_path)

    print(len(env._get_obs()))
    print(env._get_obs().shape)
    
    import time
    while True:
        time.sleep(1)
        env.render()
        env.step(np.ones(28))
        # env.step(np.random.uniform(13)*0.8 - 0.4)

# %%

