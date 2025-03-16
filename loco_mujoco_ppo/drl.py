from traj_data import TrajData
from agent import Agent
import gymnasium as gym
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from environment import DeepMimicGymEnv

class DRL:
    def __init__(self, load_models=False):
        self.envs = DeepMimicGymEnv()
        self.n_envs = 1
        self.n_steps = 256
        self.n_obs = self.envs.env.info.observation_space.shape[0]
        self.n_actions = self.envs.env.info.action_space.shape[0]

        # self.envs = gym.vector.SyncVectorEnv([lambda: DeepMimicGymEnv() for _ in range(self.n_envs)])


        self.traj_data = TrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=self.n_actions) # 1 action choice is made
        self.agent = Agent(self.n_obs, n_actions=self.n_actions, load_models=load_models)  # 2 action choices are available
        self.optimizer = Adam(self.agent.parameters(), lr=1e-3)
        self.writer = SummaryWriter(log_dir=f'runs/{self.agent.name}')


    def rollout(self, i, render):

        obs = self.envs.reset()
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad():
                actions, probs = self.agent.get_action(obs)
            log_probs = probs.log_prob(actions)
            # next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            # done = done | truncated  # episode doesnt truncate till t = 500, so never
            try:
                next_obs, rewards, done, _ = self.envs.step(actions.numpy())
            except:
                print("unstable caught")
                self.envs.reset()
                next_obs, rewards, done, _ = self.envs.step(actions.numpy())

            # print("done: " + str(done))
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            obs = torch.Tensor(next_obs)
            if render:
                self.envs.render()
                print(self.agent.get_loss(self.traj_data))

        self.traj_data.calc_returns()

        self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        self.writer.flush()


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 200

        for i in range(epochs):

            loss = self.agent.get_loss(self.traj_data)
            if i == 0 or i == epochs - 1:
                print(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.traj_data.detach()