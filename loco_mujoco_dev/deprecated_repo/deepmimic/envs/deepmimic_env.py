# deepmimic/envs/deepmimic_env.py
import numpy as np
from loco_mujoco.environments.humanoids import HumanoidTorque
from deepmimic.envs.reference_motion import ReferenceMotion
from deepmimic.utils.reward import DeepMimicReward

class DeepMimicEnv:
    """Environment wrapper for DeepMimic-style imitation learning."""
    
    def __init__(self, env_name, reference_motion_path, w_pose=0.7, w_vel=0.3):
        """
        Initialize DeepMimic environment.
        
        Args:
            env_name: Name of the LocoMujoco environment ('HumanoidTorque')
            reference_motion_path: Path to reference motion data file
            w_pose: Weight for pose matching component in reward function
            w_vel: Weight for velocity matching component in reward function
        """
        # Create base environment (HumanoidTorque)
        self.env = HumanoidTorque(use_foot_forces=True)
        
        # Set up reference motion handler and reward function
        self.reference_motion = ReferenceMotion(self.env, reference_motion_path)
        self.reward_func = DeepMimicReward(self.env, self.reference_motion, w_pose=w_pose, w_vel=w_vel)
        
    def reset(self):
        """Reset environment and reference motion."""
        obs = self.env.reset()
        self.reward_func.reset_state()
        return obs
    
    def step(self, action):
        """Step environment with given action."""
        next_obs, _, done, info = self.env.step(action)
        
        # Compute imitation reward based on current state and reference state
        reward = self.reward_func(next_obs, action=None, next_state=next_obs, absorbing=done)
        
        return next_obs, reward, done, info
    
    def render(self):
        """Render environment."""
        return self.env.render()
