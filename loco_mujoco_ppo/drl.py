from traj_data import TrajData
from agent import Agent
import gymnasium as gym
import torch
from torch import nn
from torch.optim import Adam
from environment import DeepMimicGymEnv
import numpy as np
from loco_mujoco.environments.humanoids import HumanoidTorque
import time

class DRL:
    def __init__(self, load_models=False):
        self.n_envs = 1
        self.envs = DeepMimicGymEnv()
        self.example = HumanoidTorque.generate(dataset_type='perfect', random_start=True)
        # self.envs = gym.vector.SyncVectorEnv([lambda: DeepMimicGymEnv() for _ in range(self.n_envs)])
        # self.n_steps = 2048
        # self.n_steps = 512
        # self.n_steps = 256
        self.n_steps = 100
        # self.n_steps = 50
        # self.n_steps = 10
        self.n_obs = self.envs.env.info.observation_space.shape[0]
        self.n_actions = self.envs.env.info.action_space.shape[0]
        mean_state = np.mean(self.envs.env.create_dataset()['states'])
        std_state = np.std(self.envs.env.create_dataset()['states'])


        self.traj_data = TrajData(self.n_steps, self.n_obs, n_actions=self.n_actions) # 1 action choice is made
        self.agent = Agent(self.n_obs, n_actions=self.n_actions, mean_state=mean_state, std_state=std_state, load_models=load_models)  # 2 action choices are available
        self.policy_optimizer = Adam(self.agent.policy.parameters(), lr=1e-3)
        self.value_optimizer = Adam(self.agent.value.parameters(), lr=1e-3)
        # self.writer = SummaryWriter(log_dir=f'runs/{self.agent.name}')


    def rollout(self, i, render):

        obs = self.envs.reset()
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            sample = self.envs.env.trajectories.get_current_sample()
            self.example.set_sim_state(sample)
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad():
                action, probs = self.agent.get_action(obs)
            log_probs = probs.log_prob(action)
            # next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            # done = done | truncated  # episode doesnt truncate till t = 500, so never
            try:
                next_obs, rewards, done, _ = self.envs.step(action.numpy())
            except:
                print("unstable caught")
                break


            # print("done: " + str(done))
            # print("obs: " + str(obs.shape))
            # print("actions: " + str(actions.shape))
            # print("rewards: " + str(rewards.shape))
            # print("lod probs: " + str(log_probs.shape))
            # print("done: " + str(done.shape))
            self.traj_data.store(t, obs[:], action, rewards, log_probs, done)
            obs = torch.Tensor(next_obs)

            if render:
                self.example.render()
                self.envs.render()
                # print("loss: " + str(self.agent.get_loss(self.traj_data).item()))
                print("done: " + str(done))
                print("reward: " + str(rewards.item()))
                print("action: " + str(action))
                time.sleep(0.1)

        # print("loss: " + str(self.agent.get_loss(self.traj_data).item()))
        print("reward: " + str(self.traj_data.rewards.mean().item()))

        self.traj_data.calc_returns()

        # self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        # self.writer.flush()


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 100

        for i in range(epochs):

            loss = self.agent.get_loss(self.traj_data, i == 0)

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
            if i == 0 or i == epochs - 1:
                print("loss: " + str(loss.item()))
        self.traj_data.detach()