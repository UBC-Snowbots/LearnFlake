from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class ObsAdapter(ABC):
    @abstractmethod
    def __call__(self, raw_obs) -> Dict[str, np.ndarray]:
        """Convert a raw observation into canonical dict form."""
        raise NotImplementedError
