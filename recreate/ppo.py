"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, policy, critic, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        self._init_hyperparameters(hyperparameters)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.env = env
        
        self.obs_dim = env.state_dim
        self.act_dim = env.action_dim

        self.actor = policy.to(self.device)                                                  # ALG STEP 1
        self.critic = critic.to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.clr)
        # self.actor_optim = SGD(self.actor.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-4)
        # self.critic_optim = SGD(self.critic.parameters(), lr=self.clr, momentum=0.9, weight_decay=0)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.cov_start)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],
            'entropy': [],
            'lr': 0,
            'clr': 0,
            'cov': self.cov_start
        }

    def get_lr(self, step, total_steps, start_lr=4e-4, end_lr=1e-5):
        return start_lr + (end_lr - start_lr) * (step / total_steps)


    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")

        t_so_far = 0 
        i_so_far = 0 
        while t_so_far < total_timesteps:
            # Update Covariance
            if self.interp_cov and t_so_far <= total_timesteps / 4 + self.timesteps_per_batch * 2:
                cov = max(self.cov_min, self.cov_min + (self.cov_start - self.cov_min) * (1 - t_so_far / total_timesteps * 4))
                self.logger['cov'] = cov
                self.cov_mat = torch.diag(torch.full(size=(self.act_dim,), fill_value=cov)).to(self.device)

            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
            
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()   
            
            t_so_far += np.sum(batch_lens)

            i_so_far += 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []
            closs = []
            ent = []
            cont = True

            for it in range(self.n_updates_per_iteration):
                new_lr = self.get_lr(t_so_far, total_timesteps, self.lr, self.end_lr)

                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                new_clr = self.get_lr(t_so_far, total_timesteps, self.clr, self.end_clr)
                self.critic_optim.param_groups[0]["lr"] = new_clr
                self.logger['lr'] = new_lr
                self.logger['clr'] = new_clr

                np.random.shuffle(inds)
                batch_n = 0
                for start in range(0, step, minibatch_size):
                    batch_n += 1
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = F.mse_loss(V, mini_rtgs)

                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss                    
                    
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                    closs.append(critic_loss.detach())
                    ent.append(entropy_loss.detach())

                    if approx_kl > 1.5 * self.target_kl:
                        cont = False
                        print(f"Early stopping at step {it}, batch {batch_n} due to KL: {approx_kl}")
                        break

                if not cont:
                    break

            avg_loss = sum(loss) / len(loss)
            avg_closs = sum(closs) / len(closs)
            avg_ent = sum(ent) / len(ent)
            self.logger['actor_losses'].append(avg_loss)
            self.logger['critic_losses'].append(avg_closs)
            self.logger['entropy'].append(avg_ent)

            self._log_summary(t_so_far, total_timesteps)

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ckpt/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ckpt/ppo_critic.pth')

            if i_so_far % self.ckpt_freq == 0:
                torch.save(self.actor.state_dict(), f'./ckpt-walk/ppo_actor_{i_so_far}.pth')
                torch.save(self.critic.state_dict(), f'./ckpt-walk/ppo_critic_{i_so_far}.pth')

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = [] 
            last_advantage = 0 

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage) 

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)


    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0
        resets = 0

        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                with torch.no_grad():
                    action, log_prob = self.get_action(obs)
                    val = self.critic(obs)

                good, new_obs, rew, done = self.env.step(action)

                if not good:
                    resets += 1
                    print("RESET", t, resets, resets / (t + 1) * 100)
                    break

                ep_dones.append(done)
                batch_obs.append(obs)
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                obs = new_obs

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.stack(batch_acts)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten().to(self.device)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals,batch_dones

    def get_action(self, obs, sample=True):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        if self.deterministic or not sample:
            return mean.detach(), 1

        return action.detach(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        self.timesteps_per_batch = 4096
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 10
        self.lr = 1e-4
        self.end_lr = 1e-4
        self.clr = 2e-4
        self.end_clr = 2e-4
        self.gamma = 0.95 
        self.clip = 0.2
        self.lam = 0.95 
        self.num_minibatches = 16
        self.ent_coef = 0 
        self.target_kl = 1
        self.max_grad_norm = 0.5 

        self.interp_cov = True
        self.cov_start = 0.05
        self.cov_min = 0.01


        self.render = False
        self.save_freq = 10
        self.ckpt_freq = 250
        self.deterministic = False
        self.seed = 42

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
        
        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self, t, tt):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = round(self.logger['lr'], 7)
        clr = round(self.logger['clr'], 7)
        cov = round(self.logger['cov'], 4)
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_rews = np.mean([item for sl in self.logger["batch_rews"] for item in sl])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean().cpu() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean().cpu() for losses in self.logger['critic_losses']])
        avg_entropy = np.mean([losses.float().mean().cpu() for losses in self.logger['entropy']])

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 5))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))
        avg_entropy = str(round(avg_entropy, 5))

        output = (
            f"\n"
            f"-------------------- Iteration #{i_so_far} --------------------\n"
            f"Total Timesteps: {t} / {tt} | {round(t / tt * 100, 2)}%\n"
            f"Average Episodic Length: {avg_ep_lens}\n"
            f"Average Action Return: {avg_rews}\n"
            f"Average Episodic Return: {avg_ep_rews}\n"
            f"Average Loss: {avg_actor_loss}\n"
            f"Average Value Loss: {avg_critic_loss}\n"
            f"Average Entropy: {avg_entropy}\n"
            f"Iteration took: {delta_t} secs\n"
            f"Learning rate: {lr}\n"
            f"Critic Learning rate: {clr}\n"
            f"Covariance: {cov}\n"
            f"------------------------------------------------------\n"
        )
        print(output, flush=True)
        with open("run.txt", "a") as f:
            f.write(f"\nIteration {i_so_far}:\tEp Len: {avg_ep_lens}\tAction Return: {avg_rews}\tEp Return: {avg_ep_rews}\tALoss: {avg_actor_loss}\tVLoss: {avg_critic_loss}\tEntropy: {avg_entropy}\tlr: {lr}\tclr: {clr}\tcov: {cov}")

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []