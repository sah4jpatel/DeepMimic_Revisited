# deepmimic/utils/visualizer.py
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

class MotionVisualizer:
    """Utility for visualizing reference and imitated motions."""
    
    def __init__(self, env, output_path=None):
        """
        Initialize visualizer.
        
        Args:
            env: DeepMimic environment
            output_path: Path to save visualizations
        """
        self.env = env
        self.output_path = output_path
        
    def record_episode(self, agent, num_steps, filename='deepmimic_visualization.mp4', fps=30):
        """
        Record an episode comparing reference and agent motion.
        
        Args:
            agent: Trained PPO agent
            num_steps: Number of steps to record
            filename: Output filename
            fps: Frames per second
        """
        # Handle output path
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, filename)
        else:
            path = filename
        
        # Store original device before moving model to CPU
        original_device = agent.device
        
        try:
            # Force model to CPU for visualization
            agent.ac = agent.ac.to('cpu')
            agent.device = torch.device('cpu')
            
            # Create environment for reference visualization
            ref_env = self.env.env.__class__(
                use_foot_forces=True, 
                random_start=False
            )
            
            # Reset agent environment
            obs = self.env.reset()
            
            # Get initial frame dimensions
            ref_env.render()
            ref_frame = ref_env.render()
            
            if ref_frame is None:
                ref_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
            height, width = ref_frame.shape[:2]
            
            # Create video writer
            video_writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width * 2, height)  # Double width for side-by-side
            )
            
            # Record frames
            print(f"Recording {num_steps} frames...")
            for step in range(num_steps):
                if step % 100 == 0:
                    print(f"Step {step}/{num_steps}")
                    
                # Get action from agent
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action = agent.ac.get_action(obs_tensor, deterministic=True)[0].cpu().numpy().squeeze()
                    
                # Step environment
                next_obs, reward, done, _ = self.env.step(action)
                
                # Get reference motion state
                ref_state = self.env.reference_motion.get_reference_state()
                
                # Create full observation for reference environment
                # Reference state has 36 dimensions, but set_sim_state expects
                # the full observation size with pelvis_tx and pelvis_ty
                obs_keys = ref_env.get_all_observation_keys()
                full_obs = np.zeros(len(obs_keys))
                full_obs[2:] = ref_state  # Skip the first two values (pelvis_tx, pelvis_ty)
                
                # Set reference state
                ref_env.set_sim_state(full_obs)
                
                # Render both environments
                ref_frame = ref_env.render()
                agent_frame = self.env.env.render()
                
                # Handle empty frames
                if ref_frame is None:
                    ref_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if agent_frame is None:
                    agent_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Ensure frames have the same dimensions
                if ref_frame.shape != agent_frame.shape:
                    target_height = max(ref_frame.shape[0], agent_frame.shape[0])
                    target_width = max(ref_frame.shape[1], agent_frame.shape[1])
                    ref_frame = cv2.resize(ref_frame, (target_width, target_height))
                    agent_frame = cv2.resize(agent_frame, (target_width, target_height))
                
                # Combine frames side-by-side
                combined_frame = np.hstack([ref_frame, agent_frame])
                video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
                
                # Update observation
                obs = next_obs
                
                if done:
                    obs = self.env.reset()
                    
            # Release video writer
            video_writer.release()
            print(f"Visualization saved to {os.path.abspath(path)}")
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Restore model to original device
            agent.ac = agent.ac.to(original_device)
            agent.device = original_device
            
            # Recreate optimizer to ensure its states are on the correct device
            old_lr = agent.optimizer.param_groups[0]['lr']
            agent.optimizer = optim.Adam(agent.ac.parameters(), lr=old_lr)
            
            # Ensure video writer is released
            if 'video_writer' in locals():
                video_writer.release()
