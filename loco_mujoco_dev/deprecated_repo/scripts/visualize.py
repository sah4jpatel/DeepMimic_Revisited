# scripts/visualize.py
import os
import argparse
import torch
from deepmimic.envs.deepmimic_env import DeepMimicEnv
from deepmimic.ppo.ppo import PPO
from deepmimic.utils.visualizer import MotionVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepMimic Visualization')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--env_name', type=str, default='HumanoidTorque', help='Environment name')
    parser.add_argument('--ref_motion', type=str, 
                        default='/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz', 
                        help='Reference motion path')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of steps to visualize')
    parser.add_argument('--output_path', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--filename', type=str, default='deepmimic_visualization.mp4', help='Output filename')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for video')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256], help='Hidden layer dimensions')
    
    return parser.parse_args()

def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = DeepMimicEnv(
        env_name=args.env_name,
        reference_motion_path=args.ref_motion,
        w_pose=0.7,
        w_vel=0.3
    )
    
    # Create PPO agent
    print("Creating PPO agent...")
    agent = PPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dims=args.hidden_dims,
        device=torch.device("cpu")  # Force CPU for visualization
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)
    
    # Create visualizer
    visualizer = MotionVisualizer(env, output_path=args.output_path)
    
    # Record episode
    visualizer.record_episode(
        agent=agent,
        num_steps=args.num_steps,
        filename=args.filename,
        fps=args.fps
    )
    
    print(f"Visualization saved to {os.path.join(args.output_path, args.filename)}")

if __name__ == "__main__":
    main()
