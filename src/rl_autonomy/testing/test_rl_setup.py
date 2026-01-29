#!/usr/bin/env python3
"""
Quick test to verify the RL training setup works before long training runs.
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

# Import our training module
from train_lift import RoboSuiteEnv, SACAgent, ReplayBuffer

def test_environment():
    """Test that the environment works."""
    print("="*50)
    print("Testing RoboSuite Environment")
    print("="*50)
    
    env = RoboSuiteEnv(render=True)
    
    print(f"✓ Environment created")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_dim}")
    
    obs = env.reset()
    print(f"✓ Environment reset, obs shape: {obs.shape}")
    
    # Run a few random steps
    total_reward = 0
    for i in range(50):
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
        if done:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"✓ Random rollout complete, total reward: {total_reward:.2f}")
    env.close()
    print(f"✓ Environment closed\n")
    return True

def test_agent():
    """Test that the agent works."""
    print("="*50)
    print("Testing SAC Agent")
    print("="*50)
    
    obs_dim = 27
    action_dim = 7
    
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device='cpu',
    )
    print(f"✓ Agent created")
    
    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.get_action(obs, deterministic=False)
    print(f"✓ Action selection works, action shape: {action.shape}")
    
    # Test replay buffer
    buffer = ReplayBuffer(1000, obs_dim, action_dim)
    for _ in range(100):
        buffer.add(
            np.random.randn(obs_dim),
            np.random.randn(action_dim),
            np.random.randn(),
            np.random.randn(obs_dim),
            0.0
        )
    print(f"✓ Replay buffer works, size: {buffer.size}")
    
    # Test update
    metrics = agent.update(buffer, batch_size=32)
    print(f"✓ Agent update works")
    print(f"  Critic loss: {metrics['critic_loss']:.4f}")
    print(f"  Actor loss: {metrics['actor_loss']:.4f}")
    print(f"  Alpha: {metrics['alpha']:.4f}\n")
    
    return True

def test_short_training():
    """Test a very short training run."""
    print("="*50)
    print("Testing Short Training (3 episodes)")
    print("="*50)
    
    env = RoboSuiteEnv(render=True)
    
    agent = SACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device='cpu',
    )
    
    buffer = ReplayBuffer(10000, env.obs_dim, env.action_dim)
    
    for episode in range(3):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            # Random for first episode, then use agent
            if episode == 0:
                action = np.random.uniform(-1, 1, env.action_dim)
            else:
                action = agent.get_action(obs)
            
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, float(done))
            
            obs = next_obs
            episode_reward += reward
            steps += 1
            env.render()
            
            # Update after warmup
            if buffer.size > 100:
                agent.update(buffer, batch_size=64)
        
        print(f"  Episode {episode+1}: {steps} steps, reward: {episode_reward:.2f}")
    
    env.close()
    print(f"✓ Short training complete\n")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RL Training Setup Verification")
    print("="*60 + "\n")
    
    try:
        test_agent()
        test_environment()
        test_short_training()
        
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now start training with:")
        print("  python train_lift.py --train")
        print("\nOr with rendering:")
        print("  python train_lift.py --train --render")
        print("\nFor faster training with GPU:")
        print("  python train_lift.py --train --cuda")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
