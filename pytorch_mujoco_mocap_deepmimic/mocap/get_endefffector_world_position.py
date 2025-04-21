import sys
import os

folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,folder)
from customEnv import MyEnv
from mocap_util import END_EFFECTORS
import mujoco
import numpy as np
from collections import defaultdict
import json
import numpy as np

env = MyEnv("")
end_effector = defaultdict(list)
cinert = []

# with mujoco.Renderer(env.model) as renderer:
    
for i in range(env.mocap_data_len):
    qpos = env.mocap.data_config[i]
    qvel = env.mocap.data_vel[i]

    env.set_state(qpos,qvel)

    mujoco.mj_forward(env.model,env.data)
    cinert_i = np.copy(env.data.cinert).flatten()
    cinert.append(cinert_i)
    for joint in END_EFFECTORS:
        body_index = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, joint)
        # print(id)
        xpos = env.data.xpos[body_index]
        # print(xpos.shape)
        end_effector[joint].append(list(xpos))

mocap_cinert = np.array(cinert)
np.savez('mocap_cinert.npz', cinert=mocap_cinert)

# with open("end_effector.json",'w') as f:
#     json.dump(end_effector,f)
#     # print(env.data.xpos.shape)
#     # renderer.update_scene(env.data)
    
#     # renderer.render()


