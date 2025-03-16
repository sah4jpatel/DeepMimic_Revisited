import json
import math

def deepmimic_to_mjcf(deepmimic_data):
    """
    Converts DeepMimic skeleton, BodyDefs, and DrawShapeDefs to MuJoCo MJCF format.
    """
    mjcf_str = '''<mujoco model="humanoid">
    <compiler angle="degree"/>
    <option gravity="0 0 -9.81"/>
    <worldbody>'''

    # Build joint and body maps for easy lookup
    joint_map = {j["ID"]: j for j in deepmimic_data["Skeleton"]["Joints"]}
    body_map = {b["ID"]: b for b in deepmimic_data["BodyDefs"]}
    shape_map = {s["ID"]: s for s in deepmimic_data["DrawShapeDefs"]}

    for body_id, body in body_map.items():
        joint = joint_map.get(body_id, {})
        shape = shape_map.get(body_id, {})

        # Get parent info
        parent_name = "root" if joint.get("Parent", -1) == -1 else joint_map[joint["Parent"]]["Name"]

        # Position and shape properties
        pos = f'{body["AttachX"]} {body["AttachY"]} {body["AttachZ"]}'
        shape_type = body["Shape"]
        mass = body["Mass"]
        rgba = f'{shape["ColorR"]} {shape["ColorG"]} {shape["ColorB"]} {shape["ColorA"]}'

        # Convert shape to MuJoCo format
        if shape_type == "sphere":
            geom_def = f'<geom type="sphere" mass="{mass}" size="{body["Param0"]}" rgba="{rgba}"/>'
        elif shape_type == "box":
            geom_def = f'<geom type="box" mass="{mass}" size="{body["Param0"]} {body["Param1"]} {body["Param2"]}" rgba="{rgba}"/>'
        elif shape_type == "capsule":
            geom_def = f'<geom type="capsule" mass="{mass}" fromto="0 0 0 0 0 {body["Param2"]}" size="{body["Param0"]}" rgba="{rgba}"/>'
        else:
            geom_def = ""

        # Define body and joint
        mjcf_str += f'''
        <body name="{body["Name"]}" pos="{pos}">
            <joint name="{body["Name"]}_joint" type="ball"/>
            {geom_def}
        </body>
        '''

    mjcf_str += "</worldbody></mujoco>"
    return mjcf_str

def buildbody(data):
    pos = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"]}'
    return f'<body name="{data["Name"]}" pos="{pos}">'

def buildgeom(data, joint):
    shape_type = data["Shape"]
    mass = data["Mass"]
    rgba = f'{data["ColorR"]} {data["ColorG"]} {data["ColorB"]} {data["ColorA"] / 2}'
    pos = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"]}'
    if joint["Type"] == "fixed":
        pos = f'{data["AttachX"] + joint["AttachX"]} {data["AttachZ"] + joint["AttachZ"]} {data["AttachY"] + joint["AttachY"]}'

    if shape_type == "sphere":
        geom_def = f'<geom type="sphere" name="{data["Name"]}" mass="{mass}" size="{data["Param0"] / 2}" pos="{pos}" rgba="{rgba}"/>'
    elif shape_type == "box":
        geom_def = f'<geom type="box" name="{data["Name"]}" mass="{mass}" size="{data["Param0"] / 2} {data["Param2"] / 2} {data["Param1"] / 2}" pos="{pos}" rgba="{rgba}"/>'
    elif shape_type == "capsule":
        fr = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"] - data["Param1"] / 2}'
        to = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"] + data["Param1"] - data["Param1"] / 2}'
        geom_def = f'<geom type="capsule" name="{data["Name"]}" mass="{mass}" fromto="{fr} {to}" size="{data["Param0"] / 2}" rgba="{rgba}"/>'
    else:
        geom_def = ""

    return geom_def

def buildjoint(data, n=1):
    joint_type_mapping = {
        "none": "free",
        "spherical": "ball",
        "revolute": "hinge",
        "fixed": "None"
    }

    joint_type = joint_type_mapping.get(data["Type"], "hinge")  # Default to hinge if unknown

    if joint_type == "None":
        return f'<!-- {data["Name"]} is a None joint and does not require an explicit joint definition -->'

    # pos = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"]}'
    pos = f'0 0 0'
    name = data["Name"]
    
    xml = ""
    if joint_type == "ball":
        for i in range(3):
            axis = ["0"] * 3
            axis[i] = "1"
            r = str(data[f"LimLow{i}"]) + " " + str(data[f"LimHigh{i}"])
            s = f'<joint type="hinge" name="{name}_{i}" axis="{" ".join(axis)}" pos="{pos}" range="{r}" />'
            xml += tabs(n) + s

        # xml += f'{tabs(n)}<joint name="{name}" type="ball" pos="{pos}" axis="0 1 0" range="{data["LimLow0"]} {data["LimHigh0"]}" />'

    elif joint_type in ["hinge", "free"]:
        xml += f'{tabs(n)}<joint name="{name}" type="{joint_type}"'
        xml += f' pos="{pos}"'
        xml += f' axis="0 -1 0"' if joint_type == "hinge" else ""

        if "LimLow0" in data and "LimHigh0" in data and data["LimLow0"] < data["LimHigh0"]:
            xml += f' range="{data["LimLow0"]} {data["LimHigh0"]}"'

        # if "TorqueLim" in data and data["TorqueLim"] > 0:
        #     xml += f' damping="{data["TorqueLim"] * 0.1}" stiffness="{data["TorqueLim"]}"'
        xml += ' />'

    # if data["IsEndEffector"] == 1:
    #     xml += f'{tabs(n)}<site name="{data["Name"]}_end" pos="{data["AttachX"]} {data["AttachY"]} {data["AttachZ"]}" size="0.02" rgba="1 0 0 1" />'

    return xml

def buildshape(data):
    shape = data["Shape"]
    pos = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"]}'
    xml = f'<site type="{shape}" name="{data["Name"]}"'

    if shape == "sphere":
        xml += f' pos="{pos}" size="{data["Param0"] / 2}"'
    elif shape == "capsule":
        theta = f'{data["AttachThetaX"]} {data["AttachThetaZ"]} {data["AttachThetaY"]}'
        xml += f'pos="{pos}" euler="{theta}" size="{data["Param0"] / 2} {data["Param1"] / 2}"'
    
    return xml + " />"

def tabs(n):
    return "\n" + "\t" * n

def mjcf(data):
    mjcf_str = '''<mujoco model="humanoid">
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81"/>
    <asset>
        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 -0.9" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>'''
        # <geom name="x_axis" type="cylinder" fromto="0 0 0 0.2 0 0" size="0.005" rgba="1 0 0 1"/>
        # <geom name="y_axis" type="cylinder" fromto="0 0 0 0 0.2 0" size="0.005" rgba="0 1 0 1"/>
        # <geom name="z_axis" type="cylinder" fromto="0 0 0 0 0 0.2" size="0.005" rgba="0 0 1 1"/>'''

    joint_map = {j["ID"]: j for j in data["Skeleton"]["Joints"]}
    body_map = {b["ID"]: b for b in data["BodyDefs"]}
    shape_map = {s["ID"]: s for s in data["DrawShapeDefs"]}

    hier, shapes = {}, {}
    for i, joint in joint_map.items():
        p = joint["Parent"]

        if p not in hier:
            hier[p] = []

        hier[p].append(i)

    for i, shape in shape_map.items():
        p = shape["ParentJoint"]
        if shape["Name"] == joint_map[p]["Name"]:
            continue
        if p not in shapes:
            shapes[p] = []
        shapes[p].append(i)

    def buildnode(node, t):
        rs = ""
        if joint_map[node]["Type"] != "fixed":
            rs = tabs(t) + buildbody(joint_map[node])
            t += 1
        rs += tabs(t) + buildgeom(body_map[node], joint_map[node])
        rs += buildjoint(joint_map[node], t)
        if node in shapes:
            for shape in shapes[node]:
                rs += tabs(t) + buildshape(shape_map[shape])
        if node in hier:
            for child in hier[node]:
                rs += buildnode(child, t)
        if joint_map[node]["Type"] != "fixed":
            rs += tabs(t - 1) + "</body>"

        return rs

    for root in hier[-1]:
        mjcf_str += buildnode(root, 1)

    mjcf_str += tabs(1) + "</worldbody>\n<actuator>"
    for j, joint in joint_map.items():
        if joint["Type"] not in ["spherical", "revolute"]:
            continue

        name = joint["Name"]
        if joint["Type"] == "spherical":
            for i in range(3):
                mjcf_str += tabs(2) + f'<motor joint="{name}_{i}" name="{name}_{i}" ctrllimited="true" ctrlrange="-1 1" gear="{joint["TorqueLim"] / 10}" />'
        else:
            # mjcf_str += tabs(2) + f'<motor joint="{name}" name="{name}" forcerange="{-joint["TorqueLim"] * 20} {joint["TorqueLim"] * 20}" />'
            mjcf_str += tabs(2) + f'<motor joint="{name}" name="{name}" ctrllimited="true" ctrlrange="-1 1" gear="{joint["TorqueLim"] / 10}" />'


    mjcf_str += tabs(1) + "</actuator>" + "\n</mujoco>"
    return mjcf_str



import numpy as np

def preprocess_reference_motion(ref_motion):
    """Convert reference motion data into a format compatible with MuJoCo qpos."""
    
    processed_motion = []  # Store formatted frames

    for frame in ref_motion:
        # Extract values from frame
        root_pos = np.array(frame[1:4])  # Root position (3D)
        root_rot = np.array(frame[4:8])  # Root rotation (quaternion)
        
        # Joint rotations (quaternions for 3D, scalars for 1D joints)
        chest_rot = np.array(frame[8:12])
        neck_rot = np.array(frame[12:16])
        
        right_hip_rot = np.array(frame[16:20])
        right_knee_rot = np.array([frame[20]])  # 1D
        right_ankle_rot = np.array(frame[21:25])
        
        right_shoulder_rot = np.array(frame[25:29])
        right_elbow_rot = np.array([frame[29]])  # 1D
        
        left_hip_rot = np.array(frame[30:34])
        left_knee_rot = np.array([frame[34]])  # 1D
        left_ankle_rot = np.array(frame[35:39])
        
        left_shoulder_rot = np.array(frame[39:43])
        left_elbow_rot = np.array([frame[43]])  # 1D

        # Construct qpos array
        qpos = np.concatenate([
            root_pos, root_rot, 
            chest_rot, neck_rot,
            right_hip_rot, right_knee_rot, right_ankle_rot,
            right_shoulder_rot, right_elbow_rot,
            left_hip_rot, left_knee_rot, left_ankle_rot,
            left_shoulder_rot, left_elbow_rot
        ])

        processed_motion.append(qpos)

    return np.array(processed_motion)

from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quat, order="XZY"):
    """
    Converts a quaternion to 3 hinge joint angles (Euler angles).
    
    Args:
        quat (array-like): A 4D quaternion [w, x, y, z].
        order (str): The Euler rotation order, e.g., "XYZ" or "ZYX".

    Returns:
        np.ndarray: 3 hinge joint angles in radians [angle1, angle2, angle3].
    """
    # Convert quaternion to rotation object
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # SciPy uses [x, y, z, w]
    
    # Convert to Euler angles
    euler_angles = rotation.as_euler(order.lower(), degrees=False)  # Radians
    
    return euler_angles

def rotate_quat(quat):
    """
    Fixes a quaternion that is inverted along the Z-axis by applying a 180-degree rotation.
    
    Parameters:
        quat (array-like): Input quaternion [x, y, z, w]
        
    Returns:
        numpy.ndarray: Corrected quaternion [x, y, z, w]
    """
    # Define the 180-degree rotation quaternion around Z-axis
    q_180_z = np.array([1, 0, 0, 0])  # (x, y, z, w)
    
    # Convert input quaternion to a Rotation object
    r_orig = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # wxyz -> xyzw
    
    # Convert 180-degree quaternion to a Rotation object
    # r_180_z = R.from_quat(q_180_z)
    r_180_z = R.from_euler('x', 180, degrees=True)
    r_180_y = R.from_euler('y', 90, degrees=True)
    
    # Apply the rotation correction (multiply quaternions)
    r_corrected = r_orig * R.from_euler('y', 180, degrees=True)
    
    # Return corrected quaternion
    r = r_corrected.as_quat()

    return [r[3], r[0], r[1], r[2]] # xyzw -> wxyz

def negZ(arr):
    return [-1 * arr[0], -1 * arr[1], arr[2]]

def negX(arr):
    return [arr[0], -1 * arr[1], -1 * arr[2]]

def translate_joints(data):
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

    root_pos = data[1:4]  # Root position (3D)
    root_pos = [root_pos[0], root_pos[2], root_pos[1] - 0.9]
    root_rot = data[4:8]  # Root rotation (quaternion)
    # root_rot = [root_rot[0], root_rot[1], root_rot[3], root_rot[2]]
    root_rot = [root_rot[0], -root_rot[1], -root_rot[3], -root_rot[2]]
    # root_rot = rotate_quat(root_rot)

    # Joint rotations (quaternions for 3D, scalars for 1D joints)
    chest_rot = quaternion_to_euler(data[8:12])
    neck_rot = quaternion_to_euler(data[12:16])
    
    right_hip_rot = quaternion_to_euler(data[16:20])
    # right_hip_rot = [right_hip_rot[0], right_hip_rot[1], -right_hip_rot[2]]
    right_knee_rot = data[20]  # 1D
    right_ankle_rot = quaternion_to_euler(data[21:25])
    # right_ankle_rot = [right_ankle_rot[0], right_ankle_rot[1], -right_ankle_rot[2]]
    
    right_shoulder_rot = quaternion_to_euler(data[25:29])
    right_elbow_rot = data[29]  # 1D
    
    left_hip_rot = quaternion_to_euler(data[30:34])
    # left_hip_rot = [left_hip_rot[0], left_hip_rot[1], -left_hip_rot[2]]
    left_knee_rot = data[34]  # 1D
    left_ankle_rot = quaternion_to_euler(data[35:39])
    
    left_shoulder_rot = quaternion_to_euler(data[39:43])
    left_elbow_rot = data[43]  # 1D

    # return [
    #     *root_pos, 
    #     *root_rot,
    #     *chest_rot,
    #     *neck_rot,
    #     *right_shoulder_rot,
    #     right_elbow_rot,
    #     *left_shoulder_rot,
    #     left_elbow_rot,
    #     *right_hip_rot,
    #     right_knee_rot,
    #     *right_ankle_rot,
    #     *left_hip_rot,
    #     left_knee_rot,
    #     *left_ankle_rot,
    # ]

    return [
        *root_pos, 
        *root_rot,
        *chest_rot,
        *neck_rot,
        *negZ(right_shoulder_rot),
        right_elbow_rot,
        *negZ(left_shoulder_rot),
        left_elbow_rot,
        *negZ(right_hip_rot),
        right_knee_rot,
        *right_ankle_rot,
        *negZ(left_hip_rot),
        left_knee_rot,
        *left_ankle_rot,
    ]

import mujoco
import time

def motion_to_posvel(frames, model, data):
    frame_pos = []

    pos, rots, linvels, angvels = [], [], [], []
    dur = 0
    for i, frame in enumerate(frames):
        # if i < 5:
        #     continue
        # if i >= 8:
        #     break

        # dt = frame[0]
        fpos = translate_joints(frame)
        frame_pos.append(fpos)
        dur += frame[0]

        data.qpos[:] = fpos
        # model.opt.timestep = dt
        # data.time += dt
        mujoco.mj_step(model, data)

        pos.append(data.xpos[1:])
        rots.append(data.xquat[1:])
        # linvels.append(data.cvel[1:, :3])
        # angvels.append(data.cvel[1:, 3:])

        # print("Simulated", i, data.time, flush=True)

        # with mujoco.viewer.launch_passive(model, data) as viewer:
        #     while viewer.is_running():
        #         pass

    # print("shape", np.array(pos).shape, np.array(rots).shape, np.array(linvels).shape, np.array(angvels).shape)

    return {
        # "horizon": 1/16 * 30,
        "horizon": frames[0][0] * len(frame_pos),
        "qpos": frame_pos,
        "pos": pos,
        "rots": rots,
        "dur": dur
    }

    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     i = 0
    #     while viewer.is_running():
    #         # pass
    #         # # mujoco.mj_step(model, data)
    #         # # viewer.sync()

    #         data.qpos[:] = frame_pos[i]
    #         mujoco.mj_step(model, data)
    #         viewer.sync()
    #         i += 1
    #         if i >= len(frame_pos):
    #             i = 0
    #         time.sleep(frames[0][0])


def qpos_diff(a, b):
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

    total = 0


def quaternion_difference(q1, q2):
    """Computes the relative quaternion rotation from q1 to q2."""
    rot1 = R.from_quat(q1)  # Convert to Rotation object
    rot2 = R.from_quat(q2)
    
    q_diff = rot2 * rot1.inv()  # Compute relative rotation
    return q_diff.as_quat()  # Return as quaternion (x, y, z, w)


import os
import torch
def save_checkpoint(policy_model, value_model, checkpoint_dir, checkpoint_prefix="checkpoint"):
    """
    Save the policy and value models to checkpoint files, keeping only the latest 2 checkpoints.
    
    Args:
        policy_model (nn.Module): The policy network model.
        value_model (nn.Module): The value network model.
        checkpoint_dir (str): Directory where checkpoints will be saved.
        checkpoint_prefix (str): Prefix for checkpoint filenames (e.g. 'checkpoint_epoch_').
    """
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Generate filenames for the checkpoint
    checkpoint_path_policy = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_policy.pth")
    checkpoint_path_value = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_value.pth")
    
    # Save the models' state_dicts
    torch.save(policy_model.state_dict(), checkpoint_path_policy)
    torch.save(value_model.state_dict(), checkpoint_path_value)
    print(f"Saved checkpoint: {checkpoint_path_policy} and {checkpoint_path_value}")

    # List all checkpoint files in the directory and sort by modification time (oldest to newest)
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x))
    )
    
    # Keep only the two latest checkpoints, remove the oldest if there are more than 2
    if len(checkpoint_files) > 4:
        # Find the oldest checkpoint and delete it
        os.remove(os.path.join(checkpoint_dir, checkpoint_files[0]))
        os.remove(os.path.join(checkpoint_dir, checkpoint_files[1]))

def load_checkpoint(policy_model, value_model, checkpoint_dir, checkpoint_prefix="checkpoint"):
    """
    Load the latest checkpoint for the policy and value models.
    
    Args:
        policy_model (nn.Module): The policy network model.
        value_model (nn.Module): The value network model.
        checkpoint_dir (str): Directory where checkpoints are saved.
        checkpoint_prefix (str): Prefix for checkpoint filenames (e.g. 'checkpoint_epoch_').
    """
    # Generate filenames for the checkpoint files
    checkpoint_path_policy = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_policy.pth")
    checkpoint_path_value = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_value.pth")
    
    # Load the model state_dicts from the checkpoint files
    if os.path.exists(checkpoint_path_policy) and os.path.exists(checkpoint_path_value):
        # Load the state dicts into the models
        policy_model.load_state_dict(torch.load(checkpoint_path_policy))
        value_model.load_state_dict(torch.load(checkpoint_path_value))
        print(f"Loaded checkpoint: {checkpoint_path_policy} and {checkpoint_path_value}")
    else:
        print(f"Checkpoint files not found: {checkpoint_path_policy} or {checkpoint_path_value}")
    
    return policy_model, value_model