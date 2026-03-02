import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

# ============================================================================
# Neural Network Components
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm and GELU activation."""
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.GELU):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            linear = nn.Linear(prev_dim, h)
            # Orthogonal initialization for better convergence
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            layers.extend([linear, nn.LayerNorm(h), activation()])
            prev_dim = h
        
        final_layer = nn.Linear(prev_dim, output_dim)
        nn.init.orthogonal_(final_layer.weight, gain=0.01) # Small gain for output
        nn.init.constant_(final_layer.bias, 0.0)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    """Gaussian policy with squashed actions."""
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256, 256)):
        super().__init__()
        self.trunk = MLP(obs_dim, hidden_dims[:-1], hidden_dims[-1])
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize output layers
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0.0)
    
    def forward(self, obs):
        h = F.gelu(self.trunk(obs)) # Consistent with MLP activation
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean), None
        
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob


class DoubleCritic(nn.Module):
    """Twin Q-networks, conditioned on skill.

    Q(s, a, z) where z is a one-hot skill vector.  Conditioning the
    critic on the skill lets it learn separate value estimates per skill,
    which prevents the skill selector from collapsing when one skill
    dominates Q-values across all states.
    """
    def __init__(self, obs_dim, action_dim, num_skills=5, hidden_dims=(256, 256, 256)):
        super().__init__()
        input_dim = obs_dim + action_dim + num_skills
        self.num_skills = num_skills
        self.q1 = MLP(input_dim, hidden_dims, 1)
        self.q2 = MLP(input_dim, hidden_dims, 1)
    
    def forward(self, obs, action, skill_onehot):
        x = torch.cat([obs, action, skill_onehot], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================================
# Hierarchical Components (5 Skills in V3)
# ============================================================================

class SkillSelectorV3(nn.Module):
    """
    High-level skill selector with 5 skills:
    0: Reach   - move gripper to cube
    1: Grasp   - close gripper on cube
    2: Lift    - lift cube upward
    3: Hold    - maintain lifted position at target height
    4: Recover - re-approach after drop
    """
    NUM_SKILLS = 5
    SKILL_NAMES = ['Reach', 'Grasp', 'Lift', 'Hold', 'Recover']
    
    def __init__(self, obs_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dims, self.NUM_SKILLS)
    
    def forward(self, obs):
        return self.net(obs)
    
    def sample(self, obs, deterministic=False):
        logits = self.forward(obs)
        
        if deterministic:
            skill = logits.argmax(dim=-1)
            return skill, None
        
        dist = Categorical(logits=logits)
        skill = dist.sample()
        log_prob = dist.log_prob(skill).unsqueeze(-1)
        return skill, log_prob
    
    def log_prob(self, obs, skill):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(skill.squeeze(-1)).unsqueeze(-1)
    
    def entropy(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.entropy().unsqueeze(-1)


class SkillConditionedActorV3(nn.Module):
    """Actor conditioned on skill (6 skills in V3)."""
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    
    def __init__(self, obs_dim, action_dim, num_skills=6, hidden_dims=(256, 256, 256)):
        super().__init__()
        self.skill_embed = nn.Embedding(num_skills, 32)
        self.trunk = MLP(obs_dim + 32, hidden_dims[:-1], hidden_dims[-1])
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, obs, skill):
        skill_emb = self.skill_embed(skill)
        x = torch.cat([obs, skill_emb], dim=-1)
        h = F.relu(self.trunk(x))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs, skill, deterministic=False):
        mean, log_std = self.forward(obs, skill)
        if deterministic:
            return torch.tanh(mean), None
        
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)
        
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
