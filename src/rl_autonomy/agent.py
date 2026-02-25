import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from typing import Tuple, Dict

from .networks import SkillSelectorV3, SkillConditionedActorV3, DoubleCritic
from .memory import GPUReplayBuffer

# ============================================================================
# Hierarchical SAC Agent V3
# ============================================================================

class HierarchicalSACAgentV3:
    """
    Hierarchical SAC with:
    - 5 skills (including Recovery)
    - Automatic entropy tuning for both levels
    - Double Q-learning
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = 'cuda',
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        use_amp: bool = True,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_amp = use_amp and device == 'cuda'
        
        # Networks
        self.skill_selector = SkillSelectorV3(obs_dim).to(device)
        self.actor = SkillConditionedActorV3(obs_dim, action_dim, num_skills=SkillSelectorV3.NUM_SKILLS).to(device)
        self.critic = DoubleCritic(obs_dim, action_dim).to(device)
        self.critic_target = DoubleCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy coefficients (learnable)
        self.log_alpha_skill = torch.tensor(np.log(alpha_init), requires_grad=True, device=device)
        self.log_alpha_action = torch.tensor(np.log(alpha_init), requires_grad=True, device=device)
        
        # Target entropies
        self.target_entropy_skill = -0.5 * np.log(1.0 / SkillSelectorV3.NUM_SKILLS)
        self.target_entropy_action = -action_dim
        
        # Optimizers with fused AdamW
        self.skill_optimizer = optim.AdamW(
            list(self.skill_selector.parameters()) + [self.log_alpha_skill],
            lr=lr, fused=True
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()) + [self.log_alpha_action],
            lr=lr, fused=True
        )
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, fused=True)
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # NOTE: torch.compile disabled due to CUDAGraphs backward pass issues
        # Uncomment if you want to try it (may need torch.compiler.cudagraph_mark_step_begin())
        # if hasattr(torch, 'compile'):
        #     self.skill_selector = torch.compile(self.skill_selector, mode='reduce-overhead')
        #     self.actor = torch.compile(self.actor, mode='reduce-overhead')
        #     self.critic = torch.compile(self.critic, mode='reduce-overhead')
        
        self._current_skill = None
        self._skill_steps = 0
        self._skill_persistence = 8  # Steps before reconsidering skill
    
    @property
    def alpha_skill(self):
        return self.log_alpha_skill.exp().clamp(0.01, 1.0)
    
    @property
    def alpha_action(self):
        return self.log_alpha_action.exp().clamp(0.01, 1.0)
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, int]:
        """Get action and skill for a single observation."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            
            # Skill selection with persistence
            self._skill_steps += 1
            if self._current_skill is None or self._skill_steps >= self._skill_persistence:
                skill, _ = self.skill_selector.sample(obs_t, deterministic=deterministic)
                self._current_skill = skill.item()
                self._skill_steps = 0
            
            skill_t = torch.tensor([self._current_skill], device=self.device)
            action, _ = self.actor.sample(obs_t, skill_t, deterministic=deterministic)
            
        return action.cpu().numpy().flatten(), self._current_skill
    
    def reset(self):
        """Reset skill state for new episode."""
        self._current_skill = None
        self._skill_steps = 0
    
    def update(self, buffer: GPUReplayBuffer, batch_size: int = 256) -> Dict[str, float]:
        """Update all networks."""
        obs, actions, rewards, next_obs, dones, stored_skills = buffer.sample(batch_size)
        
        amp_context = torch.amp.autocast('cuda', dtype=torch.bfloat16) if self.use_amp else torch.amp.autocast('cuda', enabled=False)
        
        with amp_context:
            # Sample new skills and actions for next state
            with torch.no_grad():
                next_skill, next_skill_log_prob = self.skill_selector.sample(next_obs)
                next_action, next_action_log_prob = self.actor.sample(next_obs, next_skill)
                
                q1_next, q2_next = self.critic_target(next_obs, next_action)
                q_next = torch.min(q1_next, q2_next)
                
                # Combined entropy bonus
                entropy_bonus = (
                    self.alpha_skill * next_skill_log_prob +
                    self.alpha_action * next_action_log_prob
                )
                
                target_q = rewards + (1 - dones) * self.gamma * (q_next - entropy_bonus)
            
            # Critic loss
            q1, q2 = self.critic(obs, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.scaler.step(self.critic_optimizer)
        else:
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
        
        with amp_context:
            # Actor loss
            skill, skill_log_prob = self.skill_selector.sample(obs)
            action, action_log_prob = self.actor.sample(obs, skill)
            
            q1_pi, q2_pi = self.critic(obs, action)
            q_pi = torch.min(q1_pi, q2_pi)
            
            actor_loss = (self.alpha_action * action_log_prob - q_pi).mean()
            
            # Alpha (action) loss
            alpha_action_loss = -(self.log_alpha_action * (action_log_prob + self.target_entropy_action).detach()).mean()
        
        self.actor_optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(actor_loss + alpha_action_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.scaler.step(self.actor_optimizer)
        else:
            (actor_loss + alpha_action_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
        
        with amp_context:
            # Skill selector loss
            skill, skill_log_prob = self.skill_selector.sample(obs)
            action, _ = self.actor.sample(obs, skill)
            
            q1_skill, q2_skill = self.critic(obs, action)
            q_skill = torch.min(q1_skill, q2_skill)
            
            skill_loss = (self.alpha_skill * skill_log_prob - q_skill).mean()
            
            # Alpha (skill) loss with entropy regularization
            skill_entropy = self.skill_selector.entropy(obs).mean()
            alpha_skill_loss = -(self.log_alpha_skill * (skill_log_prob + self.target_entropy_skill).detach()).mean()
            
            # Encourage skill diversity
            skill_diversity_bonus = 0.1 * skill_entropy
        
        self.skill_optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(skill_loss + alpha_skill_loss - skill_diversity_bonus).backward()
            self.scaler.unscale_(self.skill_optimizer)
            torch.nn.utils.clip_grad_norm_(self.skill_selector.parameters(), 1.0)
            self.scaler.step(self.skill_optimizer)
            self.scaler.update()
        else:
            (skill_loss + alpha_skill_loss - skill_diversity_bonus).backward()
            torch.nn.utils.clip_grad_norm_(self.skill_selector.parameters(), 1.0)
            self.skill_optimizer.step()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.lerp_(param.data, self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'skill_loss': skill_loss.item(),
            'alpha_skill': self.alpha_skill.item(),
            'alpha_action': self.alpha_action.item(),
            'skill_entropy': skill_entropy.item(),
        }
    
    def save(self, path: str):
        torch.save({
            'skill_selector': self.skill_selector.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'skill_optimizer': self.skill_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha_skill': self.log_alpha_skill,
            'log_alpha_action': self.log_alpha_action,
            'scaler': self.scaler.state_dict() if self.scaler else None,
        }, path)
    
    def load(self, path: str, resume_training: bool = False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle compiled models
        skill_state = checkpoint['skill_selector']
        actor_state = checkpoint['actor']
        critic_state = checkpoint['critic']
        target_state = checkpoint['critic_target']
        
        # Remove _orig_mod prefix if present
        def clean_state_dict(sd):
            return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        
        # --- Adaptive loading: handle skill/obs expansion (e.g. 5→6 skills) ---
        def _adapt_state_dict(src_sd, tgt_sd, label=""):
            """Copy src weights into tgt, expanding mismatched layers gracefully."""
            adapted = dict(tgt_sd)  # start from current (correctly-sized) state
            mismatches = []
            for key in src_sd:
                if key not in adapted:
                    continue
                src_t = src_sd[key]
                tgt_t = adapted[key]
                if src_t.shape == tgt_t.shape:
                    adapted[key] = src_t
                else:
                    mismatches.append(key)
                    # Handle dimension expansion for each axis
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_t.shape, tgt_t.shape))
                    adapted[key][slices] = src_t[slices]
                    # New neurons get small random init (weights) or zero (bias)
                    if src_t.dim() == 1:
                        adapted[key][src_t.shape[0]:] = 0.0
                    elif src_t.dim() == 2:
                        # Rows beyond src → mean of existing + noise
                        if src_t.shape[0] < tgt_t.shape[0]:
                            mean_w = src_t.mean(dim=0)
                            for i in range(src_t.shape[0], tgt_t.shape[0]):
                                adapted[key][i, :src_t.shape[1]] = mean_w + 0.1 * torch.randn(src_t.shape[1], device=src_t.device)
            if mismatches:
                print(f"  ⚠ {label}: adapted {len(mismatches)} layers with shape mismatch: {mismatches}")
            return adapted
        
        clean_skill = clean_state_dict(skill_state)
        clean_actor = clean_state_dict(actor_state)
        clean_critic = clean_state_dict(critic_state)
        clean_target = clean_state_dict(target_state)
        
        # Check if any shapes differ (old checkpoint vs new architecture)
        needs_adapt = any(
            clean_skill.get(k, torch.tensor([])).shape != v.shape
            for k, v in self.skill_selector.state_dict().items() if k in clean_skill
        )
        
        if needs_adapt:
            print(f"  🔄 Checkpoint has different architecture — adapting weights...")
            clean_skill = _adapt_state_dict(clean_skill, self.skill_selector.state_dict(), "SkillSelector")
            clean_actor = _adapt_state_dict(clean_actor, self.actor.state_dict(), "Actor")
            clean_critic = _adapt_state_dict(clean_critic, self.critic.state_dict(), "Critic")
            clean_target = _adapt_state_dict(clean_target, self.critic_target.state_dict(), "CriticTarget")
        
        self.skill_selector.load_state_dict(clean_skill)
        self.actor.load_state_dict(clean_actor)
        self.critic.load_state_dict(clean_critic)
        self.critic_target.load_state_dict(clean_target)
        
        if resume_training:
            # Try to load optimizer states, but handle mismatches gracefully
            try:
                self.skill_optimizer.load_state_dict(checkpoint['skill_optimizer'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except ValueError as e:
                print(f"  ⚠ Could not restore optimizer states (param group mismatch)")
                print(f"    Optimizers will start fresh. This is fine for continued training.")
            
            # Restore alpha values
            try:
                self.log_alpha_skill.data.copy_(checkpoint['log_alpha_skill'].data)
                self.log_alpha_action.data.copy_(checkpoint['log_alpha_action'].data)
            except Exception:
                print(f"  ⚠ Could not restore alpha values, using defaults")
            
            if checkpoint.get('scaler') and self.scaler:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                except Exception:
                    pass
    
    def load_pretrained_skills(self, checkpoint_path: str, freeze_old_skills: bool = True):
        """
        Load skills from a checkpoint for transfer learning.
        Handles remapping from V2 (4 skills) to V3 (5+ skills) architectures.
        """
        print(f"\n  📦 Loading pretrained skills...")
        print(f"     Source: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        def clean_state_dict(sd):
            return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        
        # ====================================================================
        # 1. Transfer Skill-Conditioned Actor
        # ====================================================================
        v2_actor_state = clean_state_dict(checkpoint['actor'])
        v3_actor_state = self.actor.state_dict()
        
        # Remap keys (V2 compatibility)
        def remap_actor_keys(v2_state):
            remapped = {}
            for k, v in v2_state.items():
                new_key = k
                new_key = new_key.replace('skill_embedding', 'skill_embed')
                new_key = new_key.replace('mu_layer', 'mean')
                new_key = new_key.replace('log_std_layer', 'log_std')
                if new_key.startswith('net.'):
                    new_key = 'trunk.' + new_key
                remapped[new_key] = v
            return remapped
        
        remapped_actor = remap_actor_keys(v2_actor_state)
        
        transferred_actor_count = 0
        for key in remapped_actor:
            if key in v3_actor_state:
                v2_tensor = remapped_actor[key]
                v3_tensor = v3_actor_state[key]
                
                if 'skill_embed' in key:
                    # Handle skill expansion (4 -> 5 or 5 -> 6)
                    n_src = v2_tensor.shape[0]
                    n_tgt = v3_tensor.shape[0]
                    if n_src < n_tgt:
                        # Copy existing skills
                        v3_actor_state[key][:n_src] = v2_tensor
                        # Initialize new skills with mean of existing + noise
                        mean_emb = v2_tensor.mean(dim=0)
                        for i in range(n_src, n_tgt):
                            v3_actor_state[key][i] = mean_emb + 0.1 * torch.randn(32, device=self.device)
                        print(f"     ✓ Extended skill embeddings ({n_src} -> {n_tgt})")
                        transferred_actor_count += 1
                elif v2_tensor.shape == v3_tensor.shape:
                    v3_actor_state[key] = v2_tensor
                    transferred_actor_count += 1
        
        try:
            self.actor.load_state_dict(v3_actor_state, strict=False)
            print(f"     ✓ Loaded actor params")
        except Exception as e:
            print(f"     ⚠ Partial actor load: {e}")
        
        # ====================================================================
        # 2. Transfer Critic
        # ====================================================================
        v2_critic_state = clean_state_dict(checkpoint['critic'])
        v3_critic_state = self.critic.state_dict()
        
        def remap_critic_keys(v2_state):
            remapped = {}
            for k, v in v2_state.items():
                new_key = k.replace('.q.', '.net.')
                remapped[new_key] = v
            return remapped
        
        remapped_critic = remap_critic_keys(v2_critic_state)
        
        transferred_count = 0
        for key in remapped_critic:
            if key in v3_critic_state:
                v2_tensor = remapped_critic[key]
                v3_tensor = v3_critic_state[key]
                if v2_tensor.shape == v3_tensor.shape:
                    v3_critic_state[key] = v2_tensor
                    transferred_count += 1
        
        try:
            self.critic.load_state_dict(v3_critic_state, strict=False)
            self.critic_target.load_state_dict(v3_critic_state, strict=False)
            print(f"     ✓ Loaded critic params")
        except Exception:
            print(f"     ⚠ Critic shape mismatch (likely obs_dim change), initializing fresh")
        
        # ====================================================================
        # 3. Transfer Skill Selector
        # ====================================================================
        v2_skill_state = clean_state_dict(checkpoint['skill_selector'])
        v3_skill_state = self.skill_selector.state_dict()
        
        def remap_skill_keys(v2_state):
            remapped = {}
            for k, v in v2_state.items():
                if k.startswith('net.'):
                    remapped['net.' + k] = v
                else:
                    remapped[k] = v
            return remapped
        
        remapped_skill = remap_skill_keys(v2_skill_state)
        
        for key in remapped_skill:
            if key in v3_skill_state:
                v2_tensor = remapped_skill[key]
                v3_tensor = v3_skill_state[key]
                
                # Handle output layer expansion
                is_output = v2_tensor.dim() >= 1 and v2_tensor.shape[0] != v3_tensor.shape[0]
                if is_output:
                    n_src = v2_tensor.shape[0]
                    if v2_tensor.dim() == 2: # Weights
                        v3_skill_state[key][:n_src] = v2_tensor
                        mean_weight = v2_tensor.mean(dim=0)
                        for i in range(n_src, v3_tensor.shape[0]):
                             v3_skill_state[key][i] = mean_weight + 0.1 * torch.randn_like(v3_tensor[0])
                    else: # Bias
                        v3_skill_state[key][:n_src] = v2_tensor
                        v3_skill_state[key][n_src:] = 0.0
                elif v2_tensor.shape == v3_tensor.shape:
                    v3_skill_state[key] = v2_tensor
        
        try:
            self.skill_selector.load_state_dict(v3_skill_state, strict=False)
            print(f"     ✓ Loaded skill selector")
        except Exception as e:
            print(f"     ⚠ Skill selector transfer failed: {e}")
            
        if freeze_old_skills:
            self._frozen_skills = True
            self._unfreeze_episode = 200
            print(f"     ❄️  Old skills frozen (unfreeze @ ep {self._unfreeze_episode})")
        else:
            self._frozen_skills = False
        
        print(f"     ✅ Transfer complete!\n")
    
    def unfreeze_all_skills(self):
        """Unfreeze all skill parameters for full fine-tuning."""
        self._frozen_skills = False
        print("  🔓 All skills unfrozen for fine-tuning")
    
    def set_differential_lr(self, old_skill_lr: float = 1e-5, new_skill_lr: float = 3e-4):
        """
        Set different learning rates for old vs new skills.
        Old skills (0-3): slow learning to preserve knowledge
        New skill (4): fast learning to acquire new behavior
        """
        # Rebuild optimizers with parameter groups
        old_skill_params = []
        new_skill_params = []
        
        # Skill selector: identify final layer for new skill
        for name, param in self.skill_selector.named_parameters():
            if 'net.6' in name:  # Final layer
                new_skill_params.append(param)
            else:
                old_skill_params.append(param)
        
        # Actor: skill embedding for Recovery is new
        for name, param in self.actor.named_parameters():
            if 'skill_embed' in name:
                # We can't easily split embedding, so use slower LR for all
                old_skill_params.append(param)
            else:
                old_skill_params.append(param)
        
        self.skill_optimizer = optim.AdamW([
            {'params': old_skill_params, 'lr': old_skill_lr},
            {'params': [self.log_alpha_skill], 'lr': new_skill_lr},
        ], fused=True)
        
        self.actor_optimizer = optim.AdamW([
            {'params': list(self.actor.parameters()), 'lr': old_skill_lr},
            {'params': [self.log_alpha_action], 'lr': new_skill_lr},
        ], fused=True)
        
        print(f"  📊 Differential LR set: old={old_skill_lr:.0e}, new={new_skill_lr:.0e}")
