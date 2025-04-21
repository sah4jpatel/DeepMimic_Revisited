#!/usr/bin/env python3

# Modification based on https://github.com/mingfeisun/DeepMimic_mujoco/
import os
import json
import math
import copy
import numpy as np
from os import getcwd
from pyquaternion import Quaternion
import sys
folder_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(folder_path)
import torch

from mocap.mocap_util import align_position, align_rotation
from mocap.mocap_util import BODY_JOINTS, BODY_JOINTS_IN_DP_ORDER, DOF_DEF, BODY_DEFS

from transformations import euler_from_quaternion, quaternion_from_euler
from PIL import Image

# CINERT_PATH = "/Users/xiaowenma/GT/Classes/CS 8803 DRL/project/deepmimic_pytorch/test_mujoco/mujoco copy/mocap_cinert.npz"

class MocapDM(object):
    def __init__(self):
        self.num_bodies = len(BODY_DEFS)
        self.pos_dim = 3
        self.rot_dim = 4
        self.n_actions = 28
        self.n_obs = 160+69+1

    def load_mocap(self, filepath):
        self.read_raw_data(filepath)
        self.convert_raw_data()
        # self.load_cinert()
        self.calc_mean_std()
    
    # def load_cinert(self):
    #     self.cinert = np.load(CINERT_PATH)['cinert']
    #     # print(self.cinert.shape)
        

    def calc_mean_std(self):
        self.pos_mean =  np.mean(np.array(self.data_config)[:,7:],axis=0)
        self.pos_std = np.std(np.array(self.data_config)[:,7:],axis=0)
        # print(self.pos_std.shape)

        ref_states = np.hstack((np.array(self.data_config),np.array(self.data_vel)))

        self.state_mean = np.mean(ref_states,axis=0)
        self.state_std = np.std(ref_states,axis = 0)

    def normalize_state_action(self, data):
        # print(data.shape)
        assert data.shape[0] in [28,160+69+1] or data.shape[1] in [28,160+69+1], f"wrong shape{data.shape}, can't nomarlize"
        if len(data.shape)==1:
            shape = data.shape[0]
        else:
            shape = data.shape[1]
        if shape == self.n_actions:
            a_norm = data
            a_norm = (a_norm-self.pos_mean)/(self.pos_std+1e-2)
        else: # normalizing state, state in shape(n_envs,n_obs)
            a_norm = data
            a_norm[:,:69] = (a_norm[:,:69]-self.state_mean)/(self.state_std+1e-2)
            a_norm[:,69:-1] = (a_norm[:,69:-1]-self.cinert_mean)/(self.cinert_std+1e-2)
            
            a_norm = torch.Tensor(a_norm)
            # print("max min : ",torch.max(a_norm[:,69:-1]),torch.min(a_norm[:,69:-1]))
        return a_norm

    def unormalize_state_action(self,data):
        # print(data.shape)
        assert data.shape[0] in [28,160+69+1] or data.shape[1] in [28,160+69+1], f"wrong shape{data.shape}, can't unnomarlize"

        if len(data.shape)==1:
            shape = data.shape[0]
        else:
            shape = data.shape[1]
        
        if shape == self.n_actions:
            a_norm = np.copy(data)
            a_norm = a_norm*self.pos_std+self.pos_mean
        else: # normalizing state, state in shape(n_envs,n_obs)
            a_norm = data
            a_norm[:,:69]*=self.state_std
            a_norm[:,:69]+=self.state_mean

            a_norm[:,69:-1]*=self.cinert_std
            a_norm[:,69:-1]+=self.cinert_mean
            a_norm = torch.Tensor(a_norm)
        return a_norm
             

    def read_raw_data(self, filepath):
        motions = None
        all_states = []

        durations = []

        with open(filepath, 'r') as fin:
            data = json.load(fin)
            motions = np.array(data["Frames"])
            m_shape = np.shape(motions)
            # print(motions.shape)
            self.data = np.full(m_shape, np.nan)

            total_time = 0.0
            self.dt = motions[0][0]
            for each_frame in motions:
                duration = each_frame[0]
                each_frame[0] = total_time
                total_time += duration
                durations.append(duration)

                curr_idx = 1
                offset_idx = 8
                state = {}
                state['root_pos'] = align_position(each_frame[curr_idx:curr_idx+3])
                # state['root_pos'][2] += 0.08
                state['root_rot'] = align_rotation(each_frame[curr_idx+3:offset_idx])
                for each_joint in BODY_JOINTS_IN_DP_ORDER:
                    curr_idx = offset_idx
                    dof = DOF_DEF[each_joint]
                    if dof == 1:
                        offset_idx += 1
                        state[each_joint] = each_frame[curr_idx:offset_idx]
                    elif dof == 3:
                        offset_idx += 4
                        state[each_joint] = align_rotation(each_frame[curr_idx:offset_idx])
                all_states.append(state)

        self.all_states = all_states
        self.durations = durations

    def calc_rot_vel(self, seg_0, seg_1, dura):
        q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
        q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

        # q_diff =  q_0*q_1.conjugate
        # q_diff =  q_1 * q_0.conjugate
        q_diff = q_1.conjugate*q_0 # q_0 target, q_1 curr
        axis = q_diff.axis
        angle = q_diff.angle
        
        tmp_diff = angle/dura * axis
        diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

        return diff_angular
        # q_diff = q_0*q_1.conjugate.normalised
        # angular_vel = 2*q_diff/dura
        # return [angular_vel[1],angular_vel[2],angular_vel[3]]
        # euler_target = euler_from_quaternion(np.array([seg_0[1],seg_0[2],seg_0[3],seg_0[0]]),axes='rxyz')
        # euler_curr = euler_from_quaternion(np.array([seg_1[1],seg_1[2],seg_1[3],seg_1[0]]),axes='rxyz')
        # return list((np.array(euler_target)-np.array(euler_curr))/dura)

    def convert_raw_data(self):
        self.data_vel = []
        self.data_config = []

        for k in range(len(self.all_states)):
            tmp_vel = []
            tmp_angle = []
            state = self.all_states[k]
            if k == 0:
                dura = self.durations[k]
            else:
                dura = self.durations[k-1]

            # time duration
            init_idx = 0
            offset_idx = 1
            self.data[k, init_idx:offset_idx] = dura

            # root pos
            init_idx = offset_idx
            offset_idx += 3
            self.data[k, init_idx:offset_idx] = np.array(state['root_pos'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
            tmp_angle += state['root_pos'].tolist()

            # root rot
            init_idx = offset_idx
            offset_idx += 4
            self.data[k, init_idx:offset_idx] = np.array(state['root_rot'])
            if k == 0:
                tmp_vel += [0.0, 0.0, 0.0]
            else:
                tmp_vel += self.calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
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
                        tmp_vel += ((self.data[k, init_idx:offset_idx] - self.data[k-1, init_idx:offset_idx])*1.0/dura).tolist()
                    tmp_angle += state[each_joint].tolist()
                elif DOF_DEF[each_joint] == 3:
                    assert 4 == len(tmp_val)
                    offset_idx += 4
                    self.data[k, init_idx:offset_idx] = state[each_joint]
                    if k == 0:
                        tmp_vel += [0.0, 0.0, 0.0]
                    else:
                        tmp_vel += self.calc_rot_vel(self.data[k, init_idx:offset_idx], self.data[k-1, init_idx:offset_idx], dura)
                    quat = state[each_joint]
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                    
                    euler_tuple = euler_from_quaternion(quat, axes='rxyz')
                    tmp_angle += list(euler_tuple)

            self.data_vel.append(np.array(tmp_vel))
            self.data_config.append(np.array(tmp_angle))

    def play(self, mocap_filepath):
        # from mujoco_py import load_model_from_xml, MjSim, MjViewer
        import mujoco

        curr_path = getcwd()
        # xmlpath = '/mujoco/humanoid_deepmimic/envs/asset/dp_env_v2.xml'
        xmlpath = curr_path+'/test.xml'
        # with open(curr_path + xmlpath) as fin:
        #     MODEL_XML = fin.read()

        # model = load_model_from_xml(MODEL_XML)
        # sim = MjSim(model)
        # viewer = MjViewer(sim)
        model = mujoco.MjModel.from_xml_path(xmlpath)
        print(model.nbody,model.njnt,model.ngeom)
        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        print(joint_names)

        data = mujoco.MjData(model)

        self.read_raw_data(mocap_filepath)
        self.convert_raw_data()

        from time import sleep

        phase_offset = np.array([0.0, 0.0, 0.0])
        saved_frames = []
        with mujoco.Renderer(model) as renderer:
            # while True:
            for k in range(len(self.data)):
                tmp_val = self.data_config[k]
                # print(tmp_val.shape)
                data.qpos[:] = tmp_val[:]
                
                data.qpos[:3]+=phase_offset[:] # what is phase_offset
                # sim_state = sim.get_state()
                # sim_state.qpos[:] = tmp_val[:]
                # sim_state.qpos[:3] +=  phase_offset[:]
                # sim.set_state(sim_state)
                # sim.forward()
                # viewer.render()
                mujoco.mj_forward(model,data)
                renderer.update_scene(data)
                pixels = renderer.render()
                saved_frames.append(pixels)

            # sim_state = sim.get_state()
            sim_state = data
            phase_offset = sim_state.qpos[:3]
            phase_offset[2] = 0
        self.write_video(saved_frames)

    def write_video(self,frames:np.ndarray):
        curr_path = getcwd()
        
        for i,frame in enumerate(frames):
            outpath = os.path.join('/',curr_path,f"test_frame/walk/{i}.png")
            im = Image.fromarray(frame)
            im.save(outpath)


if __name__ == "__main__":
    test = MocapDM()
    curr_path = getcwd()
    test.play(curr_path + "/walk.txt")