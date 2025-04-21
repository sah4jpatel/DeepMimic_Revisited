import torch
from PPOAgent import PPOAgent
from torch.optim import Adam
import gymnasium as gym
from customEnv import MyEnv
from TrajData import TrajData
# from ReplayBuffer import ReplayBuffer
from tqdm import tqdm
import numpy as np
from mocap.mocap import MocapDM
import mujoco
from copy import deepcopy

from gymnasium.envs.registration import register

N = 2048
OBS_DIM = 35+34+160+1
ACT_DIM = 28


MOCAP = MocapDM()
MOCAP.load_mocap("walk_long.txt")
# print(len(MOCAP.data_config))
ref_states = np.hstack((np.array(MOCAP.data_config),np.array(MOCAP.data_vel)))
# print(ref_states.shape)
states_mean = np.mean(ref_states,axis = 0)
# print(states_mean.shape)
states_std = np.std(ref_states,axis=0)
# print(states_std.shape)

class DRL:
    def __init__(self):

        self.n_envs = 32
        self.n_steps = 64 # ~2 sec in mocap
        self.n_obs = 35+34+160+1 # qpos+qvel+cinert+1(phase)
        self.n_actions = 28 # 28 actuators, each action modeled as a gaussian

        self.obs_running_mean = torch.tensor(states_mean)
        self.obs_running_var = torch.tensor(states_std)

        # self.total_cnt = 0

        self.envs = gym.vector.SyncVectorEnv([lambda: MyEnv("test.xml") for _ in range(self.n_envs)])
        # self.env = gym.make("MyCustomEnv-v0")

        # self.replay_buffer = ReplayBuffer()
        self.traj_data = TrajData(self.n_steps,self.n_envs,self.n_obs,self.n_actions, 0,0) # placeholder for ind, maybe no longer in use
  
        self.agent = PPOAgent(self.n_obs, n_actions=self.n_actions, a_lambda=.95, gamma = .99)  
        # self.optimizer = Adam(self.agent.parameters(), lr=1e-3)
        self.actor_optimizer = Adam(list(self.agent.policy.parameters())+list(self.agent.policy_out.parameters()), lr=5e-5)
        self.critic_optimizer = Adam(list(self.agent.value.parameters())+list(self.agent.value_out.parameters()),lr=1e-4)

        # self.actor_optimizer = torch.optim.SGD(list(self.agent.policy.parameters())+list(self.agent.policy_out.parameters()), lr=5e-5, momentum=0.9)
        # self.critic_optimizer = torch.optim.SGD(list(self.agent.value.parameters())+list(self.agent.value_out.parameters()), lr=1e-3, momentum=0.9)


    def calc_gaes(self,gamma = 0.99, a_lambda=0.95):

        T = self.traj_data.n_steps

        self.advantages = torch.zeros_like(self.traj_data.rewards)
        self.values = torch.zeros_like(self.traj_data.rewards)

        self.values[-1] = self.agent.value_forward(self.traj_data.states[-1]).flatten()
        # print(self.values[-1].shape)
        # calc advantages
        gae = torch.zeros_like(self.values[-1])
        for t in range(T-2,-1,-1):

          value = self.agent.value_forward(self.traj_data.states[t]).flatten()
          self.values[t] = value

          next_value = self.values[t+1]*(self.traj_data.not_dones[t])
          delta = self.traj_data.rewards[t]+gamma*next_value-value
          # print(traj_data.not_dones[t])
          gae = delta + gamma*a_lambda*self.traj_data.not_dones[t]*gae
          self.advantages[t] = gae

        self.advantages = self.advantages * self.traj_data.not_dones # normalization done in DRL.py before update


    def rollout(self, i):

        obs, _ = self.envs.reset() # obs, reset_info
        obs = torch.Tensor(obs)
        buffer_ets = [-1]*self.n_envs
        # active_envs = torch.ones(self.n_envs, dtype=torch.bool)  

        for t in range(self.n_steps):

            with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
                actions, probs = self.agent.get_action(obs)
            # print(actions.shape)
            log_probs = probs.log_prob(actions).sum(-1)

            next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            # self.traj_data.store(t, obs_norm, actions, rewards, log_probs, done)
            for i in range(self.n_envs):
                if done[i] and buffer_ets[i]==-1:
                    buffer_ets[i]=t

            for i in range(self.n_envs):
                tt = buffer_ets[i]
                # print(tt)
                if tt!=-1:
                    self.traj_data.not_dones[tt:,i] = 0
            
            obs = torch.Tensor(next_obs)

        self.traj_data.calc_returns()
        self.calc_gaes()

        self.avg_reward = (self.traj_data.rewards*self.traj_data.not_dones).mean()
        self.avg_steps = np.mean((np.array(buffer_ets)+64)%64)


    def get_avg_loss(self):
        return self.avg_policy_loss,self.avg_value_loss
    
    def get_avg_reward(self):
        return self.avg_reward
    
    def get_avg_sim_steps(self):
        return self.avg_sim_steps


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 10 if self.agent.name == 'PPO' else 1
        epoch_policy_loss = []
        epoch_value_loss = []

        for _ in range(epochs):

            # policy_loss,value_loss = self.agent.get_loss(self.traj_data,self.obs_running_mean,torch.sqrt(self.obs_running_var))
            policy_loss,value_loss = self.agent.get_loss(self.traj_data,self.obs_running_mean,self.obs_running_var)
            # print(f"policy loss: {policy_loss.item()}, value loss: {value_loss.item()}")
            epoch_policy_loss.append(policy_loss.item())
            epoch_value_loss.append(value_loss.item())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph = True)
            policy_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        self.traj_data.detach()
        self.avg_policy_loss = np.mean(epoch_policy_loss)
        self.avg_value_loss = np.mean(epoch_value_loss)

    def update3(self):
        epochs = 10 
        epoch_policy_loss = []
        epoch_value_loss = []
        # batch_indices = np.random.randint(0,2048,256)

        T, N = self.traj_data.states.shape[:2]


        # # normalize advantages
        # adv_std,adv_mean = torch.std_mean(flat_advantages)
        # flat_advantages = (flat_advantages-adv_mean)/(adv_std+1e-8)

        for e in range(epochs):
            # print(f"epoch {e}")
            self.calc_gaes()
            flat_states = self.traj_data.states.reshape(T * N, -1)
            flat_actions = self.traj_data.actions.reshape(T * N, -1)
            flat_log_probs = self.traj_data.log_probs.reshape(T * N)
            flat_rewards = self.traj_data.rewards.reshape(T*N)
            flat_returns = self.traj_data.returns.reshape(T * N)
            flat_advantages = self.advantages.reshape(T * N)
            flat_values = self.values.reshape(T*N)

            flat_not_dones = self.traj_data.not_dones.reshape(T*N)

            valid_indices = torch.where(flat_not_dones)[0]

            valid_adv = flat_advantages[valid_indices]
            # print(valid_adv.shape)
            flat_advantages[valid_indices] = (flat_advantages[valid_indices]-torch.mean(valid_adv))/(torch.std(valid_adv)+1e-8)
            flat_advantages = torch.clamp(flat_advantages,-4,4)

            # print("Adv range:", flat_advantages.min().item(), flat_advantages.max().item())
            # print("Return range:", flat_returns.min().item(), flat_returns.max().item())


            batch_size = 256
            indices = torch.randperm(T * N)
            sub_vloss = []
            sub_ploss= []

            for i in range(0, T * N, batch_size):
                batch_idx = indices[i:i+batch_size]
                
                batch_states = flat_states[batch_idx]
                batch_actions = flat_actions[batch_idx]
                batch_rewards = flat_rewards[batch_idx]
                batch_returns = flat_returns[batch_idx]
                batch_advantages = flat_advantages[batch_idx]
                batch_log_probs = flat_log_probs[batch_idx]
                batch_values = flat_values[batch_idx]
                batch_not_dones = flat_not_dones[batch_idx]
                # print(batch_returns.shape,batch_values.shape)

                policy_loss,value_loss = self.agent.get_loss3(batch_states,batch_actions,batch_rewards,batch_returns,batch_advantages,batch_values,batch_log_probs,batch_not_dones)
                epoch_policy_loss.append(policy_loss.item())
                epoch_value_loss.append(value_loss.item())
                sub_ploss.append(policy_loss)
                sub_vloss.append(value_loss)
            value_loss = torch.stack(sub_vloss).mean()
            policy_loss = torch.stack(sub_ploss).mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            value_loss.backward(retain_graph = True)
            policy_loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # print(value_loss.item(),policy_loss.item())
            
    
        self.avg_policy_loss = np.mean(epoch_policy_loss)
        self.avg_value_loss = np.mean(epoch_value_loss)

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    # register(
    #     id="MyCustomEnv-v0",
    #     entry_point="env_creator:create_env",
    # )

    # env_specs = gym.envs.registry.keys()

    # Initialize the DRL agent
    
    drl = DRL()

    for i in range(1):
        drl.rollout(i)
        drl.update3()
    # print(drl.env.unwrapped.mocap.data_config.shape)

# if __name__=="__main__":

#     drl = DRL()
#     drl.rollout(0)
#     # print(drl.traj_data.states.shape)
#     drl.update()
