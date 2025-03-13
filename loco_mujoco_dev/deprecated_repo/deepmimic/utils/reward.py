# deepmimic/utils/reward.py
import numpy as np

class DeepMimicReward:
    """Reward function for imitation learning."""
    
    def __init__(self, env, reference_motion, w_pose=0.7, w_vel=0.3):
        """
        Initialize reward function.
        
        Args:
            env: Environment instance
            reference_motion: ReferenceMotion instance
            w_pose: Weight for pose matching component
            w_vel: Weight for velocity matching component
        """
        self.env = env
        self.reference_motion = reference_motion
        self.w_pose = w_pose
        self.w_vel = w_vel
        
    def __call__(self, state, action, next_state, absorbing):
        """Compute reward based on similarity to reference motion."""
        if absorbing:
            return 0.0
        
        ref_state = self.reference_motion.get_reference_state()
        
        # Pose reward (joint positions)
        pose_diff = np.mean(np.square(state[2:] - ref_state[2:]))
        pose_reward = np.exp(-2.0 * pose_diff)
        
        # Velocity reward (joint velocities)
        vel_diff = np.mean(np.square(state[-len(ref_state[2:]):] - ref_state[-len(ref_state[2:]):]))
        vel_reward = np.exp(-0.1 * vel_diff)
        
        # Combine rewards
        total_reward = self.w_pose * pose_reward + self.w_vel * vel_reward
        
        return total_reward
    
    def reset_state(self):
        """Reset reward function state."""
        self.reference_motion.reset()
