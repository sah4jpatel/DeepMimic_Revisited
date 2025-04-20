import fire
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
import time

import mujoco
import mujoco.viewer
import numpy as np

from utils import deepmimic_to_mjcf, mjcf, motion_to_posvel, save_checkpoint, load_checkpoint
from env import MuJoCoBackflipEnv
# from agent import PPOAgent
from ppo import PPO


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, tanh = False, init=False):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            *([nn.Tanh()] if tanh else [])
        )

        if init:
            for layer in [self.net[0], self.net[2], self.net[4]]:
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.normal_(layer.bias, mean=0, std=0.01)
    
    def forward(self, state):
        return self.net(state)
    
def load_data(file):
    with open(file, "r") as f:
        data = json.load(f)

    return data

def main(
    log_file: str = "run.txt",
    view: bool = False
):
    torch.manual_seed(14)

    motion = load_data("data/walk.txt")

    # Convert to MJCF
    # mjcf_model = mjcf(skeleton)

    # with open("data/humanoid_temp.xml", "w") as f:
    #     f.write(mjcf_model)
    
    # mj = mujoco.MjModel.from_xml_string(mjcf_model)
    mj = mujoco.MjModel.from_xml_path("data/humanoid.xml")
    mj_data = mujoco.MjData(mj)

    # qpos:
    # 0 -   2:   root positions
    # 3 -   6:   root rotation
    # 7 -   9:   chest
    # 10 -  12:  neck
    # 13 -  15:  right shoulder
    #       16:  right elbow
    # 17 -  19:  left shoulder
    #       20:  left elbow
    # 21 -  23:  right hip
    #       24:  right knee
    # 25 -  27:  right ankle
    # 28 -  30:  left hip
    #       31:  left knee
    # 32 -  34:  left ankle  

    ref = motion_to_posvel(motion["Frames"], mj, mj_data)

    # with mujoco.viewer.launch_passive(mj, mj_data) as viewer:
    #     while viewer.is_running():
    #         for _ in range(100):
    #             mj_data.qpos[0:7] = [0] * 7
    #             # mj_data.qpos[2] = -0.1
    #             mujoco.mj_step(mj, mj_data)
    #             viewer.sync()
    #         time.sleep(1/100)
    # exit()

    env = MuJoCoBackflipEnv("data/humanoid.xml", ref)
    state_dim = env.state_dim
    action_dim = env.action_dim

    policy = PolicyNetwork(state_dim, action_dim, True, True)
    value = PolicyNetwork(state_dim, 1)

    if view:
        policy.load_state_dict(torch.load("./ckpt/ppo_actor.pth"))
        value.load_state_dict(torch.load("./ckpt/ppo_critic.pth"))
        ppo = PPO(policy, value, env)

        state = env.reset(0)
        r = []
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                with torch.no_grad():   
                    action, lp = ppo.get_action(state, sample=False)

                good, next_state, reward, done = env.step(action)

                r.append(reward)

                if not good:
                    print("FAIL")
                    state = env.reset()
                    continue

                viewer.sync()

                time.sleep(1 / 30)

                state = next_state

                if done:
                    print("Restarting")
                    print("Total R", np.mean(r))
                    r = []
                    state = env.reset(0)


        exit()


    # resume training
    # policy.load_state_dict(torch.load("./ckpt/ppo_actor.pth"))
    # value.load_state_dict(torch.load("./ckpt/ppo_critic.pth"))


    from datetime import datetime
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n\n========== New Run [{dt}] ==========\n")



    ppo = PPO(policy, value, env)
    ppo.learn(20_000_000)
    exit()



if __name__ == "__main__":
    fire.Fire(main)