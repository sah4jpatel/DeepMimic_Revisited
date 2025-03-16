import fire
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm

import mujoco
import mujoco.viewer

from utils import deepmimic_to_mjcf, mjcf, motion_to_posvel, save_checkpoint, load_checkpoint
from env import MuJoCoBackflipEnv
from agent import PPOAgent


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

    # motion = load_data("data/backflip.txt")
    motion = load_data("data/walk.txt")
    skeleton = load_data("data/humanoid.txt")

    # Convert to MJCF
    mjcf_model = mjcf(skeleton)

    with open("data/humanoid.xml", "w") as f:
        f.write(mjcf_model)
    
    mj = mujoco.MjModel.from_xml_string(mjcf_model)
    # mj = mujoco.MjModel.from_xml_path("data/test.xml")
    # mj = mujoco.MjModel.from_xml_path("data/humanoid.xml")
    mj_data = mujoco.MjData(mj)
    # mj_data.qpos[:] = torch.rand(mj.nq)

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

    # print(mj_data.qpos)

    # ranges = [
    #     [-1, 1],
    #     [-1, 1],
    #     [-1, 1],

    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],

    #     [-3, 0.2],
    #     [-1, 1],
    #     [-0.5, 3],

    #     [0, 3],

    #     [-3, 0.2],
    #     [-1, 1],
    #     [-0.5, 3],

    #     [0, 3],

    #     [-1, 1],
    #     [-0.5, 0.5],
    #     [-1, 2],
        
    #     [-3, 0],
        
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-1, 1],

    #     [-1, 1],
    #     [-0.5, 0.5],
    #     [-1, 2],
        
    #     [-3, 0],
        
    #     [-0.5, 0.5],
    #     [-0.5, 0.5],
    #     [-1, 1]
    # ]

    # for i in range(14, 35 - 7):
    #     for j in range(2):
    #         mj_data.qpos[i + 7] = ranges[i][j]
    #         mj_data.qpos[i + 6] = 0

    #         with mujoco.viewer.launch_passive(mj, mj_data) as viewer:
    #             while viewer.is_running():
    #                 pass
    #                 # mujoco.mj_step(mj, mj_data)
    #                 # viewer.sync()

    # -3.14, 0.5
    # -1.5, 1.5
    # -0.7, 3.14

    # mj_data.qpos[:] = [
    #     -0.277519, -0.005175, -0.275686, 
        
    #     0.996669, 0.010241, -0.06552, -0.04748, 
        
    #     0.022436352995831216, -0.05558161328288369, 0.01032610779036241, 
        
    #     0.008585410891410728, 0.1255182919732971, -0.023491383918910393, 
        
    #     1.3133670722824653, -0.9434231149182488, 1.455538059794618, 
        
    #     0.709389, 
        
    #     -1.3667916419067616, -0.7516846536347037, -1.5840022756322623, 
        
    #     0.792293, 
        
    #     -0.13860982618898215, -1.059264358530604, -0.21801772356102495, 
        
    #     -1.559657, 
        
    #     -0.00014596418046105425, 0.14456159013226366, -0.09833377529979528, 
        
    #     -0.005763184707466212, -1.1168137733988295, 0.020430359295476836, 
        
    #     -1.572356, 
        
    #     0.00014596418046105425, 0.14456159013226366, 0.09833377529979528 
    #     ]

    # with mujoco.viewer.launch_passive(mj, mj_data) as viewer:
    #     while viewer.is_running():
    #         pass
    #         # mujoco.mj_step(mj, mj_data)
    #         # viewer.sync()




    env = MuJoCoBackflipEnv(mjcf_model, ref)
    state_dim = env.state_dim
    action_dim = env.action_dim

    policy = PolicyNetwork(state_dim, action_dim)
    value = PolicyNetwork(state_dim, 1)

    ppo = PPOAgent(policy, value)



    policy, value = load_checkpoint(policy, value, "./ckpt-walk", "ckpt_199")
    ppo.policy = policy
    ppo.value = value

    import time
    state = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            with torch.no_grad():   
                action, _ = ppo.get_action(state)

            next_state, reward, done = env.step(action)

            viewer.sync()

            print("Done", done, state[0].item(), env.data.time)

            time.sleep(1 / 40)

            state = next_state


    exit()



    episodes = 500
    samples = 4096

    all_rewards = []

    for episode in tqdm(range(episodes), desc="Episodes: "):  # Training iterations
        log_probs, rewards, returns, states, actions, next_states, dones = [], [], [], [], [], [], []
        n = 0

        # progress = tqdm(total=samples, desc="Samples: ")

        while n < samples:
            ep_r = []
            state = env.reset()
            done = False

            while not done:
                action, log_prob = ppo.get_action(state)
                next_state, reward, done = env.step(action)

                states.append(state)
                actions.append(action)
                ep_r.append(reward)
                rewards.append(reward)
                log_probs.append(log_prob)
                next_states.append(next_state)
                dones.append(1 if done else 0)
                n += 1
                # progress.update(1)

                if n >= samples:
                    break

                state = next_state
            
            rs = ep_r[:]
            for i in reversed(range(len(ep_r) - 1)):
                rs[i] += rs[i + 1] * 0.99
            
            returns.extend(rs)


        # for s in tqdm(range(samples), desc="Samples: "):  # Simulate episode
        #     if done:
        #         state = env.reset()

        #     action, log_prob = ppo.get_action(state)
        #     next_state, reward, done = env.step(action)

        #     states.append(state)
        #     actions.append(action)
        #     rewards.append(reward)
        #     log_probs.append(log_prob)
        #     next_states.append(next_state)
        #     dones.append(1 if done else 0)

        #     state = next_state
        
        # Train PPO agent
        ppo.update(states, actions, rewards, returns, log_probs, next_states, dones)

        all_rewards.append((sum(rewards), sum(rewards) / samples))

        if (episode + 1) % 1 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards) / samples}")

        if (episode + 1) % 10 == 0:
            with open("run.txt", "w") as f:
                for tuple_data in all_rewards:
                    # Convert each tuple to a string, separating by space (or any delimiter you prefer)
                    line = str(tuple_data[0]) + "\t\t" + str(tuple_data[1])
                    # Write the line to the file followed by a newline character
                    f.write(line + "\n")
            
            save_checkpoint(ppo.policy, ppo.value, "./ckpt-walk", f"ckpt_{episode}")



if __name__ == "__main__":
    fire.Fire(main)