

import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

class PPO:


    def __init__(self, policy, critic, env, **hyperparameters):
        # Initialize hyperparameters and device
        self._init_hyperparameters(hyperparameters)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Environment and dimensions
        self.env = env
        self.obs_dim = env.state_dim
        self.act_dim = env.action_dim

        # Actor and critic networks
        self.actor = policy.to(self.device)
        self.critic = critic.to(self.device)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.clr)

        # Covariance matrix for Gaussian policy
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.cov_start)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # Hierarchical imitation loss coefficient
        self.hier_coef = getattr(self, 'hier_coef', 0.01)

        # Logging structure
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy': [],
            'lr': 0,
            'clr': 0,
            'cov': self.cov_start
        }


        if self.seed is not None:
            torch.manual_seed(self.seed)

    def get_lr(self, step, total_steps, start_lr, end_lr):
        return start_lr + (end_lr - start_lr) * (step / total_steps)

    def learn(self, total_timesteps):
        print(f"Learning... {self.max_timesteps_per_episode} timesteps/episode, "
              f"{self.timesteps_per_batch} timesteps/batch for total {total_timesteps}")
        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:
            
            if self.interp_cov and t_so_far <= total_timesteps/4 + self.timesteps_per_batch*2:
                cov = max(self.cov_min,
                          self.cov_min + (self.cov_start - self.cov_min) * (1 - t_so_far / total_timesteps * 4))
                self.logger['cov'] = cov
                self.cov_mat = torch.diag(torch.full((self.act_dim,), cov)).to(self.device)


            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()


            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far


            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            total_steps = batch_obs.size(0)
            for _ in range(self.n_updates_per_iteration):
                new_lr = self.get_lr(t_so_far, total_timesteps, self.lr, self.end_lr)
                new_clr = self.get_lr(t_so_far, total_timesteps, self.clr, self.end_clr)
                self.actor_optim.param_groups[0]['lr'] = new_lr
                self.critic_optim.param_groups[0]['lr'] = new_clr
                self.logger['lr'] = new_lr
                self.logger['clr'] = new_clr

                inds = np.arange(total_steps)
                np.random.shuffle(inds)
                minibatch_size = total_steps // self.num_minibatches
                early_stop = False

                for start in range(0, total_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = inds[start:end]

                    mb_obs = batch_obs[mb_idx]
                    mb_acts = batch_acts[mb_idx]
                    mb_logp = batch_log_probs[mb_idx]
                    mb_adv = A_k[mb_idx]
                    mb_rtgs = batch_rtgs[mb_idx]


                    V, curr_log_probs, entropy = self.evaluate(mb_obs, mb_acts)
                    ratios = torch.exp(curr_log_probs - mb_logp)


                    surr1 = ratios * mb_adv
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()


                    if self.hier_coef > 0:
                        hier = self._compute_hierarchical_loss(mb_acts, mb_obs)
                        actor_loss += self.hier_coef * hier


                    critic_loss = F.mse_loss(V, mb_rtgs)


                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()


                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()


                    self.logger['actor_losses'].append(actor_loss.item())
                    self.logger['critic_losses'].append(critic_loss.item())
                    self.logger['entropy'].append(entropy.mean().item())


                    approx_kl = ((ratios - 1) - (curr_log_probs - mb_logp)).mean()
                    if approx_kl > 1.5 * self.target_kl:
                        early_stop = True
                        break
                if early_stop:
                    break


            self._log_summary(t_so_far, total_timesteps)
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ckpt/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ckpt/ppo_critic.pth')
            if i_so_far % self.ckpt_freq == 0:
                torch.save(self.actor.state_dict(), f'./ckpt_bf/ppo_actor_{i_so_far}.pth')
                torch.save(self.critic.state_dict(), f'./ckpt_bf/ppo_critic_{i_so_far}.pth')

    def _compute_hierarchical_loss(self, actions, obs):
        """
        Hierarchical L2 loss between predicted actions and reference qpos frames,
        grouping parent and child joints.
        """
        device = actions.device
        
        progress = obs[:, 0]
        num = len(self.env.ref_motion['qpos'])
        idx = (progress * num).long().clamp(0, num - 1)


        ref_all = torch.tensor(np.stack(self.env.ref_motion['qpos']),
                                dtype=torch.float, device=device)
        ref_qpos = ref_all[idx]
        ref_actions = ref_qpos[:, 7:] 
        
        hier_groups = {
            (0,1,2):    [3,4,5,6,7,8,14,15,16,21,22,23],
            (3,4,5):    [],
            (6,7,8):    [9],
            (10,11,12): [13],
            (14,15,16): [17],
            (21,22,23): [24],
            (17,):      [18,19,20],
            (24,):      [25,26,27],
        }
        loss = 0.0
        for parents, children in hier_groups.items():
            pidx = list(parents)
            pdiff = actions[:, pidx] - ref_actions[:, pidx]
            pfl = pdiff.pow(2).sum(dim=1)
            if children:
                cidx = children
                cdiff = actions[:, cidx] - ref_actions[:, cidx]
                cfl = cdiff.pow(2).sum(dim=1)
                subtree = pfl + 0.5 * cfl
            else:
                subtree = pfl
            loss += subtree.mean()
        return loss

    def rollout(self):
        batch_obs, batch_acts, batch_log_probs = [], [], []
        batch_rews, batch_lens, batch_vals, batch_dones = [], [], [], []
        t = 0
        while t < self.timesteps_per_batch:
            obs = self.env.reset()
            done = False
            ep_vals, ep_rews, ep_dones = [], [], []
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                action, logp = self.get_action(obs)
                val = self.critic(obs)
                good, new_obs, rew, done = self.env.step(action)
                if not good:
                    break
                batch_obs.append(obs)
                batch_acts.append(action)
                batch_log_probs.append(logp)
                ep_vals.append(val.flatten())
                ep_rews.append(rew)
                ep_dones.append(done)
                obs = new_obs
                if done:
                    break
            batch_lens.append(len(ep_rews))
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        return (
            torch.stack(batch_obs).to(self.device),
            torch.stack(batch_acts).to(self.device),
            torch.tensor(batch_log_probs, dtype=torch.float, device=self.device),
            batch_rews, batch_lens, batch_vals, batch_dones
        )

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages, last_adv = [], 0
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (ep_rews[t] + self.gamma * ep_vals[t+1] * (1-ep_dones[t+1]) - ep_vals[t])
                else:
                    delta = ep_rews[t] - ep_vals[t]
                last_adv = delta + self.gamma * self.lam * (1-ep_dones[t]) * last_adv
                advantages.insert(0, last_adv)
            batch_advantages.extend(advantages)
        return torch.tensor(batch_advantages, dtype=torch.float, device=self.device)

    def get_action(self, obs, sample=True):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.deterministic or not sample:
            return mean.detach(), log_prob.detach()
        return action.detach(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs, dist.entropy()

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch        = 4096
        self.max_timesteps_per_episode  = 1600
        self.n_updates_per_iteration    = 10
        self.lr                         = 1e-4
        self.end_lr                     = 1e-4
        self.clr                        = 2e-4
        self.end_clr                    = 2e-4
        self.gamma                      = 0.95
        self.clip                       = 0.2
        self.lam                        = 0.95
        self.num_minibatches            = 16
        self.ent_coef                   = 0
        self.target_kl                  = 1
        self.max_grad_norm              = 0.5

        self.interp_cov                 = True
        self.cov_start                  = 0.05
        self.cov_min                    = 0.01

        self.render                     = False
        self.save_freq                  = 10
        self.ckpt_freq                  = 250
        self.deterministic              = False
        self.seed                       = None


        self.hier_coef                  = hyperparameters.get('hier_coef', 0.0)


        for name, val in hyperparameters.items():
            setattr(self, name, val)

    def _log_summary(self, t_so_far, total_timesteps):
        delta_t = (time.time_ns() - self.logger['delta_t']) / 1e9
        self.logger['delta_t'] = time.time_ns()

        avg_lens    = np.mean(self.logger['batch_lens'])
        avg_rews    = np.mean([r for ep in self.logger['batch_rews'] for r in ep])
        avg_ep_rew  = np.mean([sum(ep) for ep in self.logger['batch_rews']])
        avg_a_loss  = np.mean(self.logger['actor_losses'])
        avg_c_loss  = np.mean(self.logger['critic_losses'])
        avg_ent     = np.mean(self.logger['entropy'])

        print(f"\n------ Iteration {self.logger['i_so_far']} ------")
        print(f"Timesteps: {t_so_far}/{total_timesteps} ({t_so_far/total_timesteps*100:.2f}%)")
        print(f"Avg Ep Len: {avg_lens:.2f} | Avg Rew: {avg_rews:.4f} | Avg Ep Rew: {avg_ep_rew:.4f}")
        print(f"Actor Loss: {avg_a_loss:.5f} | Critic Loss: {avg_c_loss:.5f} | Entropy: {avg_ent:.5f}")
        print(f"LR: {self.logger['lr']:.1e} | CLR: {self.logger['clr']:.1e} | Cov: {self.logger['cov']:.4f}")
        print(f"Iteration Time: {delta_t:.2f}s")


        with open("run.txt", "a") as f:
            f.write(f"Iter {self.logger['i_so_far']}: EpLen {avg_lens:.2f} | EpRew {avg_ep_rew:.4f}")
            f.write(f" | A_Loss {avg_a_loss:.5f} | C_Loss {avg_c_loss:.5f}\n")


        self.logger['batch_lens'].clear()
        self.logger['batch_rews'].clear()
        self.logger['actor_losses'].clear()
        self.logger['critic_losses'].clear()
        self.logger['entropy'].clear()

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

