import fire
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from env import MuJoCoBackflipEnv
from ppo import PPO
from utils import motion_to_posvel, save_checkpoint


class GATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        N = Wh.size(1)
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        e = self.leakyrelu(
            torch.matmul(torch.cat([Wh_i, Wh_j], dim=-1), self.a).squeeze(-1)
        )
        mask = adj.unsqueeze(0) > 0
        e = e.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        h_prime = torch.matmul(alpha, Wh)
        return F.elu(h_prime)


def build_adj(env):
    parents = env.model.body_parentid
    mapping = {i: i-2 for i in range(2, env.model.nbody)}
    N = env.model.nbody - 2
    adj = torch.zeros(N, N, device=env.device)
    for i in range(2, env.model.nbody):
        p = int(parents[i])
        if p >= 2:
            u, v = mapping[i], mapping[p]
            adj[u, v] = 1
            adj[v, u] = 1
    adj += torch.eye(N, device=env.device)
    return adj


class GATPolicyNetwork(nn.Module):
    def __init__(self, env, in_dim, hidden_dim, action_dim):
        super(GATPolicyNetwork, self).__init__()
        self.device = env.device
        self.adj = build_adj(env)
        self.n_nodes = self.adj.size(0)
        self.input_proj = nn.Linear(3, in_dim).to(self.device)
        self.gat1 = GATLayer(in_dim, hidden_dim).to(self.device)
        self.gat2 = GATLayer(hidden_dim, hidden_dim).to(self.device)
        self.fc = nn.Sequential(
            nn.Linear(self.n_nodes * hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        ).to(self.device)

    def forward(self, state):
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True
        b = state.size(0)
        feat = state[:, 1:]
        N = self.n_nodes
        pos_start = 3
        pos_end = 3 * (N + 1)
        pos_rel = feat[:, pos_start:pos_end]  
        h = pos_rel.view(b, N, 3)             
        h = self.input_proj(h)   
        h = self.gat1(h, self.adj)
        h = self.gat2(h, self.adj)
        h_flat = h.view(b, -1)
        out = self.fc(h_flat)                
        return out.squeeze(0) if squeeze else out


def load_data(file):
    with open(file,'r') as f:
        return json.load(f)

def main(log_file: str = "run.txt", view: bool = False):
    torch.manual_seed(14)
    data = load_data("data/backflip.txt")
    mj = mujoco.MjModel.from_xml_path("data/humanoid.xml")
    mj_data = mujoco.MjData(mj)
    ref = motion_to_posvel(data["Frames"], mj, mj_data)

    env = MuJoCoBackflipEnv("data/humanoid.xml", ref)

    policy = GATPolicyNetwork(env, in_dim=4, hidden_dim=64, action_dim=env.action_dim)
    value = nn.Sequential(
        nn.Linear(env.state_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    ).to(env.device)


    if True:
        print(f"Resuming from actor_ckpt, critic_ckpt")
        policy.load_state_dict(torch.load("./ckpt/ppo_actor.pth", map_location=env.device))
        value.load_state_dict(torch.load("./ckpt/ppo_critic.pth",   map_location=env.device))

    if view:
        policy.load_state_dict(torch.load("./ckpt/ppo_actor.pth"))
        value.load_state_dict(torch.load("./ckpt/ppo_critic.pth"))
        ppo = PPO(policy, value, env)
        state = env.reset(0)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                with torch.no_grad():
                    action, _ = ppo.get_action(state, sample=False)
                good, state, rew, done = env.step(action)
                viewer.sync()
                time.sleep(1/30)
                if done or not good:
                    state = env.reset()
        return

    # training
    ppo = PPO(policy, value, env, hier_coef=0.005)
    ppo.learn(20_000_000)

if __name__ == "__main__":
    fire.Fire(main)
