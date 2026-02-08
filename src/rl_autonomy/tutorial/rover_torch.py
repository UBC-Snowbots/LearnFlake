import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributions.normal as Normal
import numpy as np

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000, 
                 n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = self.action_space.high
        self.min_action = self.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        # Create the network
        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                  fc2_dims=layer2_size, n_actions=n_actions,
                                  name='actor', learning_rate=actor_learning_rate)
        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name='critic_1', learning_rate=critic_learning_rate)
        self.critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name='critic_2', learning_rate=critic_learning_rate)

        # Create the target network
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                         fc2_dims=layer2_size, n_actions=n_actions,
                                         name='target_actor', learning_rate=actor_learning_rate)
        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name='target_critic_1', learning_rate=critic_learning_rate)
        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size,
                                      fc2_dims=layer2_size, n_actions=n_actions,
                                      name='target_critic_2', learning_rate=critic_learning_rate)
        
        self.noise = noise
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation, validation=False):
        if self.time_step < self.warmup and validation is False:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions, ))).to(self.actor.device)

    def remember(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass

    def update_network_parameters(self, tau):
        pass

    def save_models(self):
        pass

    def load_models(self):
        pass