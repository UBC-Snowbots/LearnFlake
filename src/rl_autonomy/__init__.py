from .assets.arms.really_simple_robot import ReallySimpleArm
from .assets.arms.simple_arm import SimpleArm
from .rl_agent import RobosuiteGymWrapper
from .networks import MLP, GaussianActor, DoubleCritic, SkillSelectorV3, SkillConditionedActorV3
from .memory import GPUReplayBuffer
from .agent import HierarchicalSACAgentV3
from .config import Config
from .env_wrapper import RoboSuiteEnvV3, SubprocVecEnvV3
from .trainer import train_v3, evaluate_v3
from .main import main


from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

ROBOT_CLASS_MAPPING.setdefault("SimpleArm", FixedBaseRobot)
ROBOT_CLASS_MAPPING.setdefault("ReallySimpleArm", FixedBaseRobot)

__all__ = ["SimpleArm", "ReallySimpleArm", "RobosuiteGymWrapper"]