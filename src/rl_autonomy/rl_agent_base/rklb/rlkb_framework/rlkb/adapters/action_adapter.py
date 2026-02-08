from abc import ABC, abstractmethod
import numpy as np


class ActionAdapter(ABC):
    @abstractmethod
    def __call__(self, agent_action) -> np.ndarray:
        """Convert an agent action into environment action space format."""
        raise NotImplementedError
