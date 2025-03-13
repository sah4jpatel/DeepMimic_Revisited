# deepmimic/envs/reference_motion.py
import numpy as np

class ReferenceMotion:
    """Handles reference motion data for DeepMimic-style imitation learning."""
    
    def __init__(self, env, motion_path, random_start=True):
        """
        Initialize reference motion handler.
        
        Args:
            env: LocoMujoco environment
            motion_path: Path to reference motion data file
            random_start: Whether to start from a random frame in the motion
        """
        self.env = env
        self.motion_path = motion_path
        self.random_start = random_start
        self.current_frame = 0
        
        # Load motion data
        self._load_motion_data()
        
    def _load_motion_data(self):
        """Load reference motion data from file."""
        self.motion_data = np.load(self.motion_path)
        
        # Extract states
        if 'states' in self.motion_data:
            self.states = self.motion_data['states']
            print(f"Loaded motion data with {len(self.states)} frames")
        else:
            raise ValueError("Expected 'states' in motion data file, but not found")
        
        self.trajectory_length = len(self.states)
        
    def get_reference_state(self):
        """Get the current reference state from the motion data."""
        return self.states[self.current_frame]
    
    def advance(self):
        """Advance to the next frame in the reference motion."""
        self.current_frame = (self.current_frame + 1) % self.trajectory_length
        
    def reset(self):
        """Reset the reference motion to the beginning or a random frame."""
        if self.random_start:
            self.current_frame = np.random.randint(0, self.trajectory_length)
        else:
            self.current_frame = 0
