#!/usr/bin/env python3
"""
Hierarchical SAC Training for Rover2026 Arm - Version 3 (Robust Recovery)
Main Entry Point
"""

import sys
import os
import argparse

# Ensure we can import modules from the package if run as a script
if __package__ is None:
    # Add src to path
    ROOT = os.path.dirname(os.path.abspath(__file__))
    SRC_PATH = os.path.abspath(os.path.join(ROOT, ".."))
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    from rl_autonomy.config import Config
    from rl_autonomy.trainer import train_v3, evaluate_v3
else:
    from .config import Config
    from .trainer import train_v3, evaluate_v3

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical SAC V3 - Robust Recovery Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode (CLI overrides Config.MODE)
    mode = parser.add_argument_group("Mode")
    mode.add_argument('--train', action='store_true', help='Train (or set Config.MODE="train")')
    # Default is None so we can distinguish CLI usage from Config
    mode.add_argument('--eval', type=str, default=None, 
                     help='Evaluate model (defaults to Config.EVAL_CHECKPOINT if not set)')
    mode.add_argument('--resume', type=str, default=None, 
                     help='Resume training (defaults to Config.RESUME_CHECKPOINT if not set)')
    
    # Environment
    env_args = parser.add_argument_group("Environment")
    env_args.add_argument('--domain_rand', action='store_true', default=Config.DOMAIN_RANDOMIZATION, 
                         help='Domain randomization')
    env_args.add_argument('--difficulty', type=int, default=Config.EVAL_DIFFICULTY, choices=[0, 1, 2], 
                         help='Evaluation difficulty (0=easy, 1=medium, 2=hard)')
    
    # Training (defaults from Config class)
    train_args = parser.add_argument_group("Training")
    train_args.add_argument('--episodes', type=int, default=Config.EPISODES, help='Training episodes')
    train_args.add_argument('--num_envs', type=int, default=Config.NUM_ENVS, help='Parallel environments')
    train_args.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    train_args.add_argument('--buffer_size', type=int, default=Config.BUFFER_SIZE, help='Replay buffer size')
    train_args.add_argument('--updates_per_step', type=int, default=Config.UPDATES_PER_STEP, 
                           help='Gradient updates per step')
    train_args.add_argument('--cuda', action='store_true', default=Config.USE_CUDA, help='Use CUDA')
    train_args.add_argument('--no_cuda', dest='cuda', action='store_false', help='Disable CUDA')
    
    # Transfer Learning (defaults from Config class)
    transfer_args = parser.add_argument_group("Transfer Learning")
    transfer_args.add_argument('--pretrained', type=str, default=Config.PRETRAINED_PATH,
                               help='Path to V2 checkpoint to load pre-trained skills from')
    transfer_args.add_argument('--freeze_old_skills', action='store_true', default=Config.FREEZE_OLD_SKILLS,
                               help='Freeze old skills (Reach/Grasp/Lift/Hold) initially')
    transfer_args.add_argument('--no_freeze', dest='freeze_old_skills', action='store_false',
                               help='Do not freeze old skills')
    transfer_args.add_argument('--differential_lr', action='store_true', default=Config.DIFFERENTIAL_LR,
                               help='Use different learning rates for old vs new skills')
    transfer_args.add_argument('--old_skill_lr', type=float, default=Config.OLD_SKILL_LR,
                               help='Learning rate for old skills (preservation)')
    transfer_args.add_argument('--new_skill_lr', type=float, default=Config.NEW_SKILL_LR,
                               help='Learning rate for new skills (acquisition)')
    transfer_args.add_argument('--unfreeze_episode', type=int, default=Config.UNFREEZE_EPISODE,
                               help='Episode at which to unfreeze old skills for fine-tuning')
    
    # Evaluation
    eval_args = parser.add_argument_group("Evaluation")
    eval_args.add_argument('--eval_episodes', type=int, default=Config.EVAL_EPISODES, help='Evaluation episodes')
    
    args = parser.parse_args()
    
    # Logic to resolve mode
    cli_train = args.train
    cli_eval_path = args.eval
    cli_resume_path = args.resume
    
    mode_selected = None
    
    # 1. CLI Explicit Flags
    if cli_eval_path is not None:
        args.eval = cli_eval_path
        mode_selected = 'eval'
    elif cli_resume_path is not None:
        args.resume = cli_resume_path
        mode_selected = 'resume'
    elif cli_train:
        mode_selected = 'train'
    
    # 2. Config Fallback
    if mode_selected is None:
        mode_selected = Config.MODE
        
        # Apply config paths if needed
        if mode_selected == 'eval':
            args.eval = Config.EVAL_CHECKPOINT
        elif mode_selected == 'resume':
            args.resume = Config.RESUME_CHECKPOINT

    # 3. Execution
    if mode_selected == 'eval':
        if not args.eval:
            print("Error: Evaluation mode selected but no checkpoint provided (Config.EVAL_CHECKPOINT is None?)")
            sys.exit(1)
        evaluate_v3(args)
        
    elif mode_selected == 'resume':
        if not args.resume:
            print("Error: Resume mode selected but no checkpoint provided (Config.RESUME_CHECKPOINT is None?)")
            sys.exit(1)
        train_v3(args)
        
    elif mode_selected == 'train':
        train_v3(args)
        
    else:
        print("\n  ⚠️  No valid mode specified!")
        print(f"  Config.MODE is '{Config.MODE}'")
        print("  Use --train, --eval [path], or --resume [path]")
        parser.print_help()

if __name__ == "__main__":
    main()
