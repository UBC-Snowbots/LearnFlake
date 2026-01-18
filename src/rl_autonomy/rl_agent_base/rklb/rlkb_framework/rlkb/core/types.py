from dataclasses import dataclass
from typing import Dict

import numpy as np

# Canonical observation type used across environments and agents.
Obs = Dict[str, np.ndarray]


@dataclass
class Transition:
    obs: Obs
    action: np.ndarray
    reward: float
    next_obs: Obs
    done: bool
    info: dict
