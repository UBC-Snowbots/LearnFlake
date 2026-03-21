import torch

class GPUReplayBufferHER:
    """
    GPU-native replay buffer with in-sample Future HER relabeling.

    Assumptions:
    - Desired goal is the last 3 elements of obs/next_obs.
    - Achieved goal is next_obs[-6:-3] (EEF xyz slice).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
        her_ratio: float = 0.8,
        her_success_reward: float = 100.0,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.her_ratio = float(her_ratio)
        self.her_success_reward = float(her_success_reward)

        # Pre-allocate everything on GPU
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr, 0] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)

        # 1. Grab standard batch
        obs = self.obs[idxs].clone()
        actions = self.actions[idxs].clone()
        rewards = self.rewards[idxs].clone()
        next_obs = self.next_obs[idxs].clone()
        dones = self.dones[idxs].clone()

        # 2. Create boolean mask for HER
        # squeeze() ensures it's [batch_size] and not [batch_size, 1]
        her_mask = (torch.rand((batch_size,), device=self.device) < self.her_ratio)

        # 3. Apply HER relabeling
        if her_mask.any():
            # Grab the achieved goals [batch_size, 3]
            achieved_goals = next_obs[:, -6:-3]
            
            # Using torch.where is the safest way to avoid shape mismatch errors in PyTorch
            # Expand the mask to [batch_size, 3] to match the goal slice dimensions
            mask_3d = her_mask.unsqueeze(-1).expand(-1, 3)
            
            # If mask is True, replace with achieved_goal. If False, keep original goal.
            obs[:, -3:] = torch.where(mask_3d, achieved_goals, obs[:, -3:])
            next_obs[:, -3:] = torch.where(mask_3d, achieved_goals, next_obs[:, -3:])
            
            # Expand mask to [batch_size, 1] for rewards and dones
            mask_1d = her_mask.unsqueeze(-1)
            
            # Update rewards and dones
            rewards = torch.where(mask_1d, torch.tensor(self.her_success_reward, device=self.device), rewards)
            dones = torch.where(mask_1d, torch.tensor(1.0, device=self.device), dones)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }
