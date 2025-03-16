import fire
import torch
import torch.nn as nn
import torch.optim as optim
import json

import mujoco
import mujoco.viewer

from utils import deepmimic_to_mjcf, mjcf
from env import MuJoCoBackflipEnv


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            # nn.Tanh()  # Outputs between -1 and 1 (normalized joint angles)
        )
    
    def forward(self, state):
        return self.net(state)

def load_data(file):
    with open(file, "r") as f:
        data = json.load(f)

    return data

def main(
    motion_file: str = ""
):
    torch.manual_seed(42)

    motion = load_data("data/backflip.txt")
    skeleton = load_data("data/humanoid.txt")

    # Convert to MJCF
    mjcf_model = mjcf(skeleton)

    with open("data/humanoid.xml", "w") as f:
        f.write(mjcf_model)

    # mj = mujoco.MjModel.from_xml_string(mjcf_model)
    # # mj = mujoco.MjModel.from_xml_path("data/test.xml")
    # mj_data = mujoco.MjData(mj)

    # with mujoco.viewer.launch_passive(mj, mj_data) as viewer:
    #     while viewer.is_running():
    #         # pass
    #         mujoco.mj_step(mj, mj_data)
    #         viewer.sync()


    
    env = MuJoCoBackflipEnv(mjcf_model, motion)
    state_dim = env.state_dim
    action_dim = env.action_dim

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)


if __name__ == "__main__":
    fire.Fire(main)