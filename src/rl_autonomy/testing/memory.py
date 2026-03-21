import torch

# ============================================================================
# GPU-Resident Replay Buffer
# ============================================================================

class GPUReplayBuffer:
    """GPU-resident replay buffer with pinned memory staging."""
    
    def __init__(self, capacity, obs_dim, action_dim, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate GPU tensors
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.skills = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        
        # CPU staging buffer with pinned memory
        self._buffer_size = 256
        self._buffer_ptr = 0
        self._obs_buf = torch.zeros((self._buffer_size, obs_dim), dtype=torch.float32, pin_memory=True)
        self._act_buf = torch.zeros((self._buffer_size, action_dim), dtype=torch.float32, pin_memory=True)
        self._rew_buf = torch.zeros((self._buffer_size, 1), dtype=torch.float32, pin_memory=True)
        self._next_buf = torch.zeros((self._buffer_size, obs_dim), dtype=torch.float32, pin_memory=True)
        self._done_buf = torch.zeros((self._buffer_size, 1), dtype=torch.float32, pin_memory=True)
        self._skill_buf = torch.zeros((self._buffer_size, 1), dtype=torch.long, pin_memory=True)
        
        self._stream = torch.cuda.Stream(device=device)
    
    def add(self, obs, action, reward, next_obs, done, skill=0):
        """Stage transition in CPU buffer, flush to GPU when full."""
        self._obs_buf[self._buffer_ptr] = torch.from_numpy(obs)
        self._act_buf[self._buffer_ptr] = torch.from_numpy(action)
        self._rew_buf[self._buffer_ptr, 0] = reward
        self._next_buf[self._buffer_ptr] = torch.from_numpy(next_obs)
        self._done_buf[self._buffer_ptr, 0] = float(done)
        self._skill_buf[self._buffer_ptr, 0] = skill
        self._buffer_ptr += 1
        
        if self._buffer_ptr >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Transfer staged data to GPU asynchronously."""
        if self._buffer_ptr == 0:
            return
        
        with torch.cuda.stream(self._stream):
            n = self._buffer_ptr
            end_ptr = (self.ptr + n) % self.capacity
            
            if end_ptr > self.ptr or self.ptr + n <= self.capacity:
                # Simple case: no wraparound
                idx = slice(self.ptr, self.ptr + n)
                self.obs[idx] = self._obs_buf[:n].to(self.device, non_blocking=True)
                self.actions[idx] = self._act_buf[:n].to(self.device, non_blocking=True)
                self.rewards[idx] = self._rew_buf[:n].to(self.device, non_blocking=True)
                self.next_obs[idx] = self._next_buf[:n].to(self.device, non_blocking=True)
                self.dones[idx] = self._done_buf[:n].to(self.device, non_blocking=True)
                self.skills[idx] = self._skill_buf[:n].to(self.device, non_blocking=True)
            else:
                # Wraparound case
                first_part = self.capacity - self.ptr
                self.obs[self.ptr:] = self._obs_buf[:first_part].to(self.device, non_blocking=True)
                self.obs[:end_ptr] = self._obs_buf[first_part:n].to(self.device, non_blocking=True)
                self.actions[self.ptr:] = self._act_buf[:first_part].to(self.device, non_blocking=True)
                self.actions[:end_ptr] = self._act_buf[first_part:n].to(self.device, non_blocking=True)
                self.rewards[self.ptr:] = self._rew_buf[:first_part].to(self.device, non_blocking=True)
                self.rewards[:end_ptr] = self._rew_buf[first_part:n].to(self.device, non_blocking=True)
                self.next_obs[self.ptr:] = self._next_buf[:first_part].to(self.device, non_blocking=True)
                self.next_obs[:end_ptr] = self._next_buf[first_part:n].to(self.device, non_blocking=True)
                self.dones[self.ptr:] = self._done_buf[:first_part].to(self.device, non_blocking=True)
                self.dones[:end_ptr] = self._done_buf[first_part:n].to(self.device, non_blocking=True)
                self.skills[self.ptr:] = self._skill_buf[:first_part].to(self.device, non_blocking=True)
                self.skills[:end_ptr] = self._skill_buf[first_part:n].to(self.device, non_blocking=True)
            
            self.ptr = (self.ptr + n) % self.capacity
            self.size = min(self.size + n, self.capacity)
        
        self._buffer_ptr = 0
    
    def sample(self, batch_size):
        """Sample batch directly from GPU memory."""
        self._flush_buffer()
        torch.cuda.current_stream().wait_stream(self._stream)
        
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
            self.skills[indices],
        )
    
    def __len__(self):
        return self.size
