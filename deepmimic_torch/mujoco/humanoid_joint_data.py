
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

JOINT_WEIGHT = {"root": 1, "chest": 0.5, "neck": 0.3, "right_hip": 0.5, 
                "right_knee": 0.3, "right_ankle": 0.2, "right_shoulder": 0.3, "right_elbow": 0.2, 
                "right_wrist": 0.0, "left_hip": 0.5, "left_knee": 0.3, "left_ankle": 0.2, 
                "left_shoulder": 0.3, "left_elbow": 0.2, "left_wrist": 0.0}

