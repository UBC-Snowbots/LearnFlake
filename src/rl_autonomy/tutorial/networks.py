import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")

sys.path.insert(0, ROBO_PATH)

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributions.normal as Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name='critic', chkpt_dir='tmp/rover', learning_rate=0.001):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_rover')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims) # or use pointers
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weighted_decay=0.005)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print('Created CriticNetwork on device:', self.device)
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q(action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name='actor', chkpt_dir='tmp/rover', learning_rate=0.001, max_action=1):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_rover')
        self.max_action = max_action

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        print('Created ActorNetwork on device:', self.device)
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = T.tanh(self.output(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))