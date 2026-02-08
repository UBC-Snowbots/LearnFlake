import os, sys
import json
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# --- Setup Paths ---
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

if __name__ == "__main__":

    # 1. This function now auto-discovers "default_rover2025.json"
    #    inside the library folder, just like it does for "Panda".
    controller_config = load_composite_controller_config(
        controller="BASIC", 
        robot="Rover2025"
    )

    # 2. Create Environment
    env = suite.make(
        env_name="Lift",
        robots=["Rover2025"],
        controller_configs=controller_config,
        has_renderer=True,
        control_freq=20,
        gripper_types="default",
    )
    
    # 3. Verify
    obs = env.reset()
    low, high = env.action_spec
    print(f"Loaded successfully! Action Space Size: {low.shape[0]}") 
    # Should print 8 (7 joints + 1 gripper)