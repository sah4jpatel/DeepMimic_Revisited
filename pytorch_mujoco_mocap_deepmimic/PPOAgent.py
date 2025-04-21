import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions import Normal
from TrajData import TrajData

class PPOAgent(nn.Module):
    def __init__(self, n_obs, n_actions, a_lambda, gamma=.99, epochs=10): # for this model, ? actuator
        super().__init__()
        self.name = 'PPO'
        self.epochs = epochs
        self.n_obs = n_obs
        self.n_actions = n_actions

        self.policy = nn.Sequential(
            nn.Linear(n_obs,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            # nn.Linear(512,n_actions)
        )
        self.policy_out = nn.Linear(512,n_actions)

        self.value = nn.Sequential(
            nn.Linear(n_obs,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            # nn.Linear(256,1)
        )
        self.value_out = nn.Linear(256,1)
        # TODO: Same optimizer here, original paper used 2 optimizers

        self.a_lambda = a_lambda
        self.gamma = gamma

        self.init_weight()

        # end student code

    def policy_forward(self,obs):
       out = self.policy_out(self.policy(obs))
       return out
    
    def value_forward(self,obs):
       out = self.value_out(self.value(obs))
       return out

    def init_weight(self):
        nn.init.uniform_(self.policy_out.weight, -0.003, 0.003)
        nn.init.constant_(self.policy_out.bias, 0.0)
        nn.init.uniform_(self.value_out.weight, -0.003, 0.003)
        nn.init.constant_(self.value_out.bias, 0.0)

    """
        Version 1: Use entire traj as in HW1, in the original paper, in each update step, the authors sampled from memory(traj?)
    """
    def get_loss(self, traj_data:TrajData, obs_mean, obs_std, epsilon=.2):
        policy_loss = []
        value_loss = []
        T = traj_data.n_steps

        advantages = torch.zeros_like(traj_data.rewards)
        values = torch.zeros_like(traj_data.rewards)

        values[-1] = self.value_forward(traj_data.states[-1]).flatten()
        # calc advantages
        gae = torch.zeros_like(values[-1])
        for t in range(T-2,-1,-1):
          value = self.value_forward(traj_data.states[t]).flatten()
          values[t] = value

          next_value = values[t+1]*(traj_data.not_dones[t])
          delta = traj_data.rewards[t]+self.gamma*next_value-value
          # print(traj_data.not_dones[t])
          gae = delta + self.gamma*self.a_lambda*traj_data.not_dones[t]*gae
          advantages[t] = gae

        # normalize advantages
        adv_std,adv_mean = torch.std_mean(advantages)
        advantages = (advantages-adv_mean)/(adv_std+1e-8)
        advantages = torch.clamp(advantages,-4,4)

        for t in range(T):
          # A_gae = 0
          value_loss.append((traj_data.returns[t]-values[t])**2) # need to calc value_loss before t_prime loop

          actions,probs = self.get_action(traj_data.states[t])

          # print(actions.shape)
          p = probs.log_prob(traj_data.actions[t]).sum(-1)
          # print("P: ", p.shape)
          ratio = torch.exp(p-traj_data.log_probs[t])
          policy_loss.append(torch.min(ratio*advantages[t].detach(),self.clip(ratio,epsilon)*advantages[t].detach()))
        # print(policy_loss[1].shape,policy_loss[-1].shape)
        policy_loss = -torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()
        # print(policy_loss,value_loss)

        # loss = policy_loss+value_loss  # replace
        print(value_loss.item(),policy_loss.item())
        return policy_loss,value_loss
    
    def get_loss2(self,cnts,buffer_states,buffer_actions,buffer_returns,buffer_rewards,buffer_log_probs,buffer_not_dones,batch_indices,epsilon=0.2):
       
        policy_loss = []
        value_loss = []
        advantages = torch.zeros_like(buffer_returns)
        values = torch.zeros_like(buffer_returns)

        total_cnt = 0
        for cnt in cnts:
          # partial_advantages = torch.zeros((cnt,1))
          # partial_values = torch.zeros((cnt,1))

          values[total_cnt+cnt-1] = self.value_forward(buffer_states[total_cnt+cnt-1]).flatten()
          # calc advantages
          gae = torch.zeros_like(values[total_cnt+cnt-1])
          for t in range(total_cnt+cnt-2,total_cnt-1,-1):
            value = self.value_forward(buffer_states[t]).flatten()
            values[t] = value

            next_value = values[t+1]*(buffer_not_dones[t])
            delta = buffer_rewards[t]+self.gamma*next_value-value
            gae = delta + self.gamma*self.a_lambda*buffer_not_dones[t]*gae
            advantages[t] = gae

          total_cnt+=cnt

        # normalize advantages
        adv_std,adv_mean = torch.std_mean(advantages)
        advantages = (advantages-adv_mean)/(adv_std+1e-8)


        for ind in batch_indices:
          value_loss.append((buffer_returns[ind]-values[ind])**2) # need to calc value_loss before t_prime loop

          actions,probs = self.get_action(buffer_states[ind])
          
          # print(actions.shape)
          p = probs.log_prob(buffer_actions[ind]).sum(-1)
          # print("P: ", p.shape)
          ratio = torch.exp(p-buffer_log_probs[ind])
          policy_loss.append(torch.min(ratio*advantages[ind].detach(),self.clip(ratio,epsilon)*advantages[ind].detach()))
        # print(policy_loss[1].shape,policy_loss[-1].shape)
        policy_loss = -torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()
        print(value_loss.item(),policy_loss.item())
        return policy_loss,value_loss
    

        
    def get_loss3(self,batch_states,batch_actions,batch_rewards,batch_returns,batch_advantages,batch_values,batch_log_probs,batch_not_dones,epsilon=0.2):
        
        # value_loss = torch.mean((batch_returns-batch_values)**2)
        value_loss = torch.sum((batch_returns-batch_values)**2*batch_not_dones)/(torch.sum(batch_not_dones)+1e-8)

        actions,probs = self.get_action(batch_states)
        
        # print(actions.shape)
        p = probs.log_prob(batch_actions).sum(-1)
        # print("P: ", p.shape)
        ratio = torch.exp(p-batch_log_probs)

        policy_loss = -torch.sum(torch.min(ratio*batch_advantages.detach(),self.clip(ratio,epsilon)*batch_advantages.detach())*batch_not_dones)/(torch.sum(batch_not_dones)+1e-8)

        return policy_loss,value_loss
       


    def clip(self,ratio, epsilon):
        return torch.clamp(ratio, 1-epsilon,1+epsilon)

    def get_action(self, obs):

        """TODO: update the clamp part, clamp the log_std_dev instead of std_dev"""
        # mean, log_std_dev = self.policy(obs).chunk(2, dim=-1)
        # mean = torch.tanh(mean)
        # mean = -50+100*(mean+1)/2 # suppose torque -50 - 50

        # std_dev = log_std_dev.exp().clamp(.2, 2)
        # dist = Normal(mean,std_dev)
        # action = dist.rsample()
        mean= self.policy_forward(obs)
        std_dev = torch.ones_like(mean)*0.01

        dist = Normal(mean,std_dev)
        action = dist.rsample()

        return action,dist