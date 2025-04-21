import math
import numpy as np
from pyquaternion import Quaternion

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee",
                        "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
                        "left_knee", "left_ankle", "left_shoulder", "left_elbow"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

BODY_DEFS = ["root", "chest", "neck", "right_hip", "right_knee", 
             "right_ankle", "right_shoulder", "right_elbow", "right_wrist", "left_hip", 
             "left_knee", "left_ankle", "left_shoulder", "left_elbow", "left_wrist"]

PARAMS_KP_KD = {"chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
        "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
        "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}

# PARAMS_KP_KD = {"chest": [100, 10], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30], 
#         "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50], "right_knee": [500, 50], 
#         "right_ankle": [400, 40], "left_hip": [500, 50], "left_knee": [500, 50], "left_ankle": [400, 40]}

JOINT_WEIGHT = {"root": 1, "chest": 0.5, "neck": 0.3, "right_hip": 0.5, 
                "right_knee": 0.3, "right_ankle": 0.2, "right_shoulder": 0.3, "right_elbow": 0.2, 
                "right_wrist": 0.0, "left_hip": 0.5, "left_knee": 0.3, "left_ankle": 0.2, 
                "left_shoulder": 0.3, "left_elbow": 0.2, "left_wrist": 0.0}

END_EFFECTORS = ["left_ankle","right_ankle","left_wrist","right_wrist"]

JOINT_RANGE = np.array([[-1.2,1.2],[-1.2,1.2],[-1.2,1.2],[-1,1],[-1,1],[-1,1],[-3.14,0.5],[-3.14,0.7],[-1.5,1.5],[0,3],[-0.5,-3.14],[-3.14,0.7],[-1.5,1.5],[0,3],[-1.2,1.2],[-2.57,1.57],[-1.,1.],[-3.14,0],[-1,1],[-1.57,1.57],[-1,1],[-1.2,1.2],[-2.57,1.57],[-1.,1.],[-3.14,0],[-1,1],[-1.57,1.57],[-1,1]])

# joint: [mujoco_start,mujuco_end), [pybullet_start,pybullet_end)], ordered by mujoco xml
MUJOCO_PYBULLET_MAP = {
    "root":[(0,7),(0,7)],
    "chest":[(7,10),(7,10)],
    "neck":[(10,13),(10,13)],
    "right_shoulder": [(13,16),(20,23)],
    "right_elbow":[(16,17),(23,24)],
    "left_shoulder":[(17,20),(31,34)],
    "left_elbow":[(20,21),(34,35)],
    "right_hip":[(21,24),(13,16)],
    "right_knee":[(24,25),(16,17)],
    "right_ankle":[(25,28),(17,20)],
    "left_hip":[(28,31),(24,27)],
    "left_knee":[(31,32),(27,28)],
    "left_ankle":[(32,25),(28,31)]
}


def align_rotation(rot): # return in wxyz
    q_input = Quaternion(rot[0], rot[1], rot[2], rot[3])
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
