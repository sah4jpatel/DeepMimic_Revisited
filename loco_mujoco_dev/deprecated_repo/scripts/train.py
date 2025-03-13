# scripts/train.py
import os
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from deepmimic.envs.deepmimic_env import DeepMimicEnv
from deepmimic.ppo.ppo import PPO
from deepmimic.utils.visualizer import MotionVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepMimic Training with PPO-GAE')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='HumanoidTorque', help='Environment name')
    parser.add_argument('--ref_motion', type=str, 
                       default='/home/sahaj/.local/lib/python3.10/site-packages/loco_mujoco/datasets/humanoids/perfect/humanoid_torque_walk/perfect_expert_dataset_det.npz', 
                       help='Reference motion path')
    
    # Training settings
    parser.add_argument('--total_steps', type=int, default=10000000, help='Total training steps')
    parser.add_argument('--steps_per_update', type=int, default=2048, help='Steps per policy update')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256], help='Hidden layer dimensions')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_ratio', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of PPO epochs per update')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--target_kl', type=float, default=0.01, help='Target KL divergence')
    parser.add_argument('--w_pose', type=float, default=0.7, help='Weight for pose matching')
    parser.add_argument('--w_vel', type=float, default=0.3, help='Weight for velocity matching')
    parser.add_argument('--value_coef', type=float, default=1.0, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=1, help='Updates between logging')
    parser.add_argument('--save_interval', type=int, default=10, help='Updates between saving model')
    parser.add_argument('--render_interval', type=int, default=20, help='Updates between rendering videos')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/videos", exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    print("Creating environment...")
    env = DeepMimicEnv(
        env_name=args.env_name,
        reference_motion_path=args.ref_motion,
        w_pose=args.w_pose,
        w_vel=args.w_vel
    )
    
    # Create PPO agent
    print("Creating PPO agent...")
    agent = PPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
        device=device
    )
    
    # Create visualizer
    visualizer = MotionVisualizer(env, output_path=f"{args.output_dir}/videos")
    
    # Training variables
    total_steps = 0
    num_updates = args.total_steps // args.steps_per_update
    all_rewards = []
    episode_rewards = []
    episode_lengths = []
    
    # Initialize tracking metrics
    current_episode_reward = 0
    current_episode_length = 0
    
    # Main training loop
    for update in range(1, num_updates + 1):
        print(f"Update {update}/{num_updates}")
        
        # Collect rollout
        start_time = time.time()
        buffer = agent.collect_rollout(env, args.steps_per_update)
        rollout_time = time.time() - start_time
        
        total_steps += args.steps_per_update
        
        # Calculate mean episode reward and lengths from buffer
        for i in range(len(buffer['rewards'])):
            current_episode_reward += buffer['rewards'][i]
            current_episode_length += 1
            
            if buffer['dones'][i]:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                
        # If episode is not done at end of buffer, count the partial reward
        if current_episode_length > 0:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
        mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        mean_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
        all_rewards.append(mean_reward)
        
        # Update policy with diagnostic info printed
        print("Updating policy...")
        update_start = time.time()
        metrics = agent.update(buffer)
        update_time = time.time() - update_start
        
        # Logging
        if update % args.log_interval == 0:
            print(f"Steps: {total_steps}")
            print(f"Mean reward: {mean_reward:.2f}")
            print(f"Mean episode length: {mean_length:.1f}")
            print(f"Value loss: {metrics['value_loss']:.4f}")
            print(f"Policy loss: {metrics['policy_loss']:.4f}")
            print(f"Entropy: {metrics['entropy']:.4f}")
            print(f"KL divergence: {metrics['kl']:.4f}")
            print(f"Clip fraction: {metrics['clip_fraction']:.4f}")
            print(f"Rollout time: {rollout_time:.2f}s, Update time: {update_time:.2f}s")
            print("-" * 50)
            
        # Save model
        if update % args.save_interval == 0:
            save_path = f"{args.output_dir}/models/model_{update}.pt"
            agent.save(save_path)
            print(f"Model saved to {save_path}")
            
            # Plot rewards
            plt.figure(figsize=(10, 5))
            plt.plot(all_rewards)
            plt.title('Training Rewards')
            plt.xlabel('Update')
            plt.ylabel('Mean Reward')
            plt.grid(True)
            reward_plot_path = f"{args.output_dir}/plots/rewards_{update}.png"
            plt.savefig(reward_plot_path)
            plt.close()
            print(f"Reward plot saved to {reward_plot_path}")
            
        # Record evaluation video
        if update % args.render_interval == 0:
            try:
                print("Recording evaluation video...")
                visualizer.record_episode(
                    agent=agent,
                    num_steps=500,
                    filename=f"comparison_{update}.mp4"
                )
            except Exception as e:
                print(f"Error recording video: {e}")
                import traceback
                traceback.print_exc()
            
    # Save final model
    agent.save(f"{args.output_dir}/models/model_final.pt")
    
    # Final reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Update')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/plots/rewards_final.png")
    plt.close()

if __name__ == "__main__":
    main()
