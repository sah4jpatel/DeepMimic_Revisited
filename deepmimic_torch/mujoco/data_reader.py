
#%%

import os
import json
import math
import copy
import numpy as np
from os import getcwd
from pyquaternion import Quaternion


#%%


from humanoid_joint_data import BODY_JOINTS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_DEFS
from state_ops import align_position, align_rotation, calc_rot_vel
import sys
sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/external_code')
from transformations import quaternion_from_euler, euler_from_quaternion

#%%

class DataReader(object):
    def __init__(self):
        self.dt = None #EDIT
        self.data = None
        pass

    def load_mocap(self, filepath):
        self.read_motion_data(filepath)
        self.convert_data()

    def read_motion_data(self, filepath):
        with open(filepath, 'r') as fin:
            data = json.load(fin)
        frames = np.array(data["Frames"])
        total_frames, frame_length = np.shape(frames)

        self.data = np.full((total_frames, frame_length), np.nan)
        self.dt = frames[0][0] #EDIT

        durations   = []
        all_states = []

        total_time = 0.
        for frame in frames:
            duration = frame[0]
            frame[0] = total_time
            total_time += duration
            durations.append(duration)

            curr_idx = 1
            offset_idx = 8
            state = {}
            state['root_pos'] = align_position(frame[curr_idx:curr_idx+3])
            state['root_rot'] = align_rotation(frame[curr_idx+3:offset_idx])
            for joint in BODY_JOINTS_IN_DP_ORDER:
                curr_idx = offset_idx
                dof = DOF_DEF[joint]
                if dof == 1:
                    offset_idx += 1
                    state[joint] = frame[curr_idx:offset_idx]
                elif dof == 3:
                    offset_idx += 4
                    state[joint] = align_rotation(frame[curr_idx:offset_idx])
            all_states.append(state)

        self.all_states = all_states
        self.durations = durations

        return frames
    
    def convert_data(self):
        self.data_vel = []
        self.data_qpos = []

        for k in range(len(self.all_states)):
            tmp_vel = []
            tmp_angle = []
            state = self.all_states[k]
            if k == 0:
                duration = self.durations[k]
            else:
                duration = self.durations[k-1]

            init_idx = 0
            offset_idx = 1
            self.data[k, init_idx:offset_idx] = duration

            # root pos
            init_idx = offset_idx
            offset_idx += 3
            self.data[k, init_idx:offset_idx] = np.array(state['root_pos'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/duration).tolist()
            tmp_angle += state['root_pos'].tolist()

            # root rot
            init_idx = offset_idx
            offset_idx += 4
            self.data[k, init_idx:offset_idx] = np.array(state['root_rot'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], duration)
            tmp_angle += state['root_rot'].tolist()

            for each_joint in BODY_JOINTS:
                init_idx = offset_idx
                tmp_val = state[each_joint]
                if DOF_DEF[each_joint] == 1:
                    assert 1 == len(tmp_val)
                    offset_idx += 1
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0]
                    else:
                        tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/duration).tolist()
                    tmp_angle += state[each_joint].tolist()
                elif DOF_DEF[each_joint] == 3:
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0, 0.0, 0.0]
                    else:
                        tmp_vel += calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], duration)
                    quat = state[each_joint]
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                    euler_tuple = euler_from_quaternion(quat, axes='rxyz')
                    tmp_angle += list(euler_tuple)
            self.data_vel.append(np.array(tmp_vel))
            self.data_qpos.append(np.array(tmp_angle))


    


#%%



if __name__ == "__main__":
    import os
    import numpy as np
    from mujoco import MjModel, MjData, mj_forward
    import mujoco_viewer
    import sys

    sys.path.append('/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/external_code')
    from transformations import quaternion_from_euler, euler_from_quaternion
    test = DataReader()

    curr_path = os.getcwd()
    xmlpath = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/envs/dp_env_v2.xml'
    motion_path = '/mnt/c/Users/cnikh/Projects/DeepMimic/deepmimic_torch/data/motions/humanoid3d_walk.txt'

    with open(xmlpath, 'r') as fin:
        MODEL_XML = fin.read()

    model = MjModel.from_xml_string(MODEL_XML)
    data = MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)


    test.read_motion_data(motion_path)
    test.convert_data()

    phase_offset = np.array([0.0, 0.0, 0.0])

    print(test.data_qpos[0].shape)
    print(test.data.shape)

    # while True:
    #     for k in range(len(test.data)):
    #         tmp_val = test.data_qpos[k]
    #         print(tmp_val)
    #         data.qpos[:] = tmp_val[:]
    #         data.qpos[:3] += phase_offset[:]
    #         mj_forward(model, data)
    #         viewer.render()
    #         print(k)
    #     final_pos = data.qpos[:3].copy()
    #     phase_offset = final_pos
    #     phase_offset[2] = 0.0

# %%
