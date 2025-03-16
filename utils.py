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

def buildgeom(data):
    shape_type = data["Shape"]
    mass = data["Mass"]
    rgba = f'{data["ColorR"]} {data["ColorG"]} {data["ColorB"]} {data["ColorA"] / 2}'
    pos = f'{data["AttachX"]} {data["AttachZ"]} {data["AttachY"]}'

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
        rs = tabs(t) + buildbody(joint_map[node])
        t += 1
        rs += tabs(t) + buildgeom(body_map[node])
        rs += buildjoint(joint_map[node], t)
        if node in shapes:
            for shape in shapes[node]:
                rs += tabs(t) + buildshape(shape_map[shape])
        if node in hier:
            for child in hier[node]:
                rs += buildnode(child, t)
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
                mjcf_str += tabs(2) + f'<motor joint="{name}_{i}" name="{name}_{i}" ctrllimited="true" ctrlrange="-1 1" gear="{joint["TorqueLim"] / 15}" />'
        else:
            # mjcf_str += tabs(2) + f'<motor joint="{name}" name="{name}" forcerange="{-joint["TorqueLim"] * 20} {joint["TorqueLim"] * 20}" />'
            mjcf_str += tabs(2) + f'<motor joint="{name}" name="{name}" ctrllimited="true" ctrlrange="-1 1" gear="{joint["TorqueLim"] / 15}" />'


    mjcf_str += tabs(1) + "</actuator>" + "\n</mujoco>"
    return mjcf_str

