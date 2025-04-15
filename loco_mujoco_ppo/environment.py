import time
import numpy as np
import gymnasium as gym
import mujoco
import mujoco_viewer
from loco_mujoco.environments.humanoids import HumanoidTorque
from loco_mujoco.utils.reward import RewardInterface
from mushroom_rl.utils.mujoco import ObservationType
import quaternion

class ImitationReward(RewardInterface):
    def __init__(self, env):
        self.env = env
        self.target_trajectory = env.trajectories
        self.obs_spec = env.obs_helper.observation_spec
    
    def __call__(self, state, action, next_state, absorbing):
        """
        Compute the reward.

        Args:
            state (np.ndarray): last state;
            action (np.ndarray): applied action;
            next_state (np.ndarray): current state.

        Returns:
            The reward for the current transition.

        """
        target_sample = self.target_trajectory.get_next_sample()
        if target_sample is None:
            self.target_trajectory.reset_trajectory()
            target_sample = self.trajectories.get_current_sample()
        # print("target sample: " + str(len(target_sample)))
        target_obs = np.array(self.env._create_observation(target_sample))
        # print("target obs: " + str(target_obs.shape))
        # print("obs spec: " + str(len(self.obs_spec)))
        # print("state: " + str(state.shape))

        assert target_obs.shape == state.shape

        # for key_name_ot, value in zip(self.obs_spec, target_obs):
        #     key, name, ot = key_name_ot
        #     if ot == ObservationType.JOINT_POS:
        #         # self._data.joint(name).qpos = value
        #         print('pos')
        #         # print(self._data.joint(name).qpos)
        #     elif ot == ObservationType.JOINT_VEL:
        #         # env._data.joint(name).qvel = value
        #         print('vel')
        #     elif ot == ObservationType.SITE_ROT:
        #         # self._data.site(name).xmat = value
        #         print('site rot')
        diff = target_obs - state
        reward = np.exp(-0.1 * np.linalg.norm(diff))
        # print("difs: " + str(np.square(target_obs - state)))
        # print("target: " + str(target_obs))
        # print("position norm: " + str(np.linalg.norm(diff[:15])))
        # print("velocity norm: " + str(np.linalg.norm(diff[15:])))
        # print("positions: " + str(self.env._get_observation_space()[0][:15]))
        # print("velocities: " + str(self.env._get_observation_space()[0][15:]))
        # print("reward: " + str(reward))
        return reward
        

    def reset_state(self):
        """
        Reset the state of the object.

        """
        pass


class DeepMimicGymEnv(gym.Env):
    """Fixed environment with proper state handling"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeepMimicGymEnv, self).__init__()
        self.env = HumanoidTorque.generate(dataset_type='perfect', random_start=True)
        self.env._reward_function = ImitationReward(self.env)
        self.viewer = None
        self.steps = 0
        # self.action_space = self.env.info.action_space.shape[0]

    def reset(self):
        # sample = self.env.trajectories.get_current_sample()
        # self.env.reset(self.env._create_observation(sample))
        self.env.reset()
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return self.env._create_observation(self.env.obs_helper._build_obs(self.env._data))

    def step(self, action):
        # print(self.env._data.ctrl.shape)
        # print("action stats:")
        # print(np.max(action))
        # print(np.min(action))
        # print(np.mean(action))
        obs, reward, _, _ = self.env.step(action) 
        # done = self.env._has_fallen(obs)
        done = False
        if done and self.steps != 0:
            print(f"it fell after {self.steps} steps")
            self.steps = 0
        elif not done:
            self.steps += 1
        return obs, np.array(reward), np.array(done), {}

    def render(self, speed=1.0):
        self.env.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()