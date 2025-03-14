#%%

import math
import numpy as np
from pyquaternion import Quaternion

#%%


def calc_rot_vel( seg_0, seg_1, duration):
    q_0 = Quaternion(seg_0[0], seg_0[1], seg_0[2], seg_0[3])
    q_1 = Quaternion(seg_1[0], seg_1[1], seg_1[2], seg_1[3])

    q_diff =  q_0.conjugate * q_1
    # q_diff =  q_1 * q_0.conjugate
    axis = q_diff.axis
    angle = q_diff.angle
    
    tmp_diff = angle/duration * axis
    diff_angular = [tmp_diff[0], tmp_diff[1], tmp_diff[2]]

    return diff_angular


def align_rotation(rot):
    q_input = Quaternion(*rot)
    q_align_right = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                                [0.0, 0.0, 1.0], 
                                                [0.0, -1.0, 0.0]]))
    q_align_left = Quaternion(matrix=np.array([[1.0, 0.0, 0.0], 
                                               [0.0, 0.0, -1.0], 
                                               [0.0, 1.0, 0.0]]))
    q_output = q_align_left * q_input * q_align_right
    return q_output.elements

def align_position(pos):
    assert len(pos) == 3
    left_matrix = np.array([[1.0, 0.0, 0.0], 
                            [0.0, 0.0, -1.0], 
                            [0.0, 1.0, 0.0]])
    pos_output = np.matmul(left_matrix, pos)
    return pos_output

def calc_angular_vel_from_quaternion(orien_0, orien_1, dt):
    seg0 = align_rotation(orien_0)
    seg1 = align_rotation(orien_1)

    q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
    q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

    q_diff =  q_0.conjugate * q_1
    # q_diff =  q_1 * q_0.conjugate
    axis = q_diff.axis
    angle = q_diff.angle
    
    tmp_vel = (angle * 1.0)/dt * axis
    vel_angular = [tmp_vel[0], tmp_vel[1], tmp_vel[2]]

    return vel_angular

def calc_diff_from_quaternion(orien_0, orien_1):
    seg0 = align_rotation(orien_0)
    seg1 = align_rotation(orien_1)

    q_0 = Quaternion(seg0[0], seg0[1], seg0[2], seg0[3])
    q_1 = Quaternion(seg1[0], seg1[1], seg1[2], seg1[3])

    q_diff =  q_0.conjugate * q_1
    # q_diff =  q_1 * q_0.conjugate
    angle = q_diff.angle
    return angle