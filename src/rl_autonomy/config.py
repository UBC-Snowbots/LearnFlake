class Config:
    """
    Edit these values to set your training configuration.
    Command-line arguments will override these if provided.
    """
    
    # === Mode ===
    MODE = "train"  # "train", "eval", or "resume"
    EVAL_CHECKPOINT = None  # Path to checkpoint for eval (e.g., "checkpoints_v3/20260206/best_model.pt")
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume training from
    
    # === Hardware ===
    USE_CUDA = True  # Use GPU acceleration
    NUM_ENVS = 16  # Parallel environments (adjust based on CPU cores)
    
    # === Training ===
    EPISODES = 2000  # Total training episodes
    BATCH_SIZE = 512  # Batch size for gradient updates
    BUFFER_SIZE = 500_000  # Replay buffer capacity
    UPDATES_PER_STEP = 2  # Gradient updates per environment step
    
    # === Environment ===
    DOMAIN_RANDOMIZATION = True  # Randomize cube position
    PERTURBATION_PROB = 0.02  # Chance of mid-episode perturbation (0-1)
    
    # === Transfer Learning ===
    # Set PRETRAINED_PATH to your V2 model to use pre-trained skills!
    PRETRAINED_PATH = None  # e.g., "checkpoints_v2/20260206_123456/best_model.pt"
    FREEZE_OLD_SKILLS = True  # Freeze Reach/Grasp/Lift/Hold initially
    DIFFERENTIAL_LR = True  # Use different LRs for old vs new skills
    OLD_SKILL_LR = 1e-5  # Learning rate for old skills (slow, preserve)
    NEW_SKILL_LR = 3e-4  # Learning rate for new skill (fast, learn)
    UNFREEZE_EPISODE = 200  # When to unfreeze old skills for fine-tuning
    
    # === Evaluation ===
    EVAL_EPISODES = 10  # Episodes to run during evaluation
    EVAL_DIFFICULTY = 2  # 0=easy, 1=medium, 2=hard (cube position range)
