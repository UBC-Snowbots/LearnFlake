import numpy as np

# Here we will create the RL agent

class NaiveModel(object):
    def __init__(self, dof):
        self.dof = dof

    def __call__(self, obs=None):
        return np.random.randn(self.dof)