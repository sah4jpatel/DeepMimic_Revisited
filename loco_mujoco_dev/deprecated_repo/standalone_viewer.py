# scripts/standalone_viewer.py
import os
import cv2
import torch
import numpy as np
import mujoco
from loco_mujoco.environments.humanoids import HumanoidTorque
from deepmimic.ppo.ppo import PPO
from deepmimic.envs.deepmimic_env import DeepMimicEnv

def visualize_model(model_path, ref_motion_path, env_name="HumanoidTorque", 
                    num_steps=1000, output_dir="visualizations", 
                    filename="standalone_visualization.mp4"):
    """
    Standalone function to visualize a trained model alongside reference motion.
    
    Args:
        model_path: Path to trained model
        ref_motion_path: Path to reference motion data
        env_name: Name of environment
        num_steps: Number of steps to visualize
        output_dir: Output directory
        filename: Output filename
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Load dataset
    data = np.load(ref_motion_path, allow_pickle=True)
    states = data["states"]
    print(f"Loaded reference motion with {len(states)} frames")
    
    # Create two separate environments: one for reference and one for predictions
    ref_env = HumanoidTorque(use_foot_forces=True, random_start=False)
    agent_env = DeepMimicEnv(
        env_name=env_name,
        reference_motion_path=ref_motion_path,
        w_pose=0.7,
        w_vel=0.3
    )
    
    # Load model
    agent = PPO(
        obs_dim=agent_env.obs_dim,
        action_dim=agent_env.action_dim,
        hidden_dims=[256, 256],
        device=torch.device('cpu')  # Force CPU for visualization
    )
    agent.load(model_path)
    
    # Get observation structure
    obs_keys = ref_env.get_all_observation_keys()
    
    # Initialize video writer
    fps = 30
    try:
        # Get reference frame dimensions by rendering once
        ref_env.render()
        frame = ref_env.render()
        if frame is None:
            # Create a default frame if rendering returns None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width * 2, height)  # Double width for side-by-side
        )
        
        # Reset agent environment
        agent_obs = agent_env.reset()
        
        print(f"Recording {num_steps} frames...")
        for step in range(min(len(states), num_steps)):
            if step % 100 == 0:
                print(f"Step {step}/{num_steps}")
            
            # 1. Render reference motion
            # Create full observation vector with zeros for pelvis x,y
            ref_full_obs = np.zeros(len(obs_keys))
            ref_full_obs[2:] = states[step]
            
            # Set state in reference environment
            ref_env.set_sim_state(ref_full_obs)
            mujoco.mj_forward(ref_env._model, ref_env._data)
            
            # Get reference frame
            ref_frame = ref_env.render()
            
            # 2. Get agent action and step environment
            obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0)
            with torch.no_grad():
                action = agent.ac.get_action(obs_tensor, deterministic=True)[0].cpu().numpy().squeeze()
            
            # Step agent environment
            agent_obs, reward, done, _ = agent_env.step(action)
            
            # Get agent frame
            agent_frame = agent_env.env.render()
            
            # 3. Combine frames and write to video
            if ref_frame is not None and agent_frame is not None:
                # Make sure both frames have the same dimensions
                if ref_frame.shape != agent_frame.shape:
                    # Resize to match
                    target_height = max(ref_frame.shape[0], agent_frame.shape[0])
                    target_width = max(ref_frame.shape[1], agent_frame.shape[1])
                    if ref_frame.shape != (target_height, target_width, 3):
                        ref_frame = cv2.resize(ref_frame, (target_width, target_height))
                    if agent_frame.shape != (target_height, target_width, 3):
                        agent_frame = cv2.resize(agent_frame, (target_width, target_height))
                
                # Combine frames horizontally
                combined_frame = np.hstack([ref_frame, agent_frame])
                video_writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            
            if done:
                agent_obs = agent_env.reset()
        
        # Release resources
        video_writer.release()
        print(f"Visualization saved to {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'video_writer' in locals():
            video_writer.release()
        ref_env.stop()
        agent_env.close()

if __name__ == "__main__":
    # Example usage
    visualize_model(
        model_path="results/models/model_20.pt",
        ref_motion_path="/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz"
    )
