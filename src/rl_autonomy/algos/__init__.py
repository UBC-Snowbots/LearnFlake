"""rl_autonomy.algos — training algorithms.

Public API:
    RLPDSAC          -- main training algorithm (TRACKER §3.1).
    BCPretrain       -- behavior-cloning trainer (gated; demos skipped for v1).
    ResidualActor    -- frozen base actor + learnable residual (gated; v1.1).
    IKExpert         -- the M1 DLS-Jacobian expert as an interactive policy
                        (TRACKER §35 — queryable for DAgger / demo generation).
"""
from .networks import Actor, Critic, EnsembleCritic
from .replay_buffer import ReplayBuffer, SymmetricReplayBuffer
from .rlpd_sac import RLPDSAC, RLPDConfig
from .bc_pretrain import BCPretrain
from .residual_actor import ResidualActor
from .expert_ik import IKExpert, ik_step

__all__ = [
    "Actor",
    "Critic",
    "EnsembleCritic",
    "ReplayBuffer",
    "SymmetricReplayBuffer",
    "RLPDSAC",
    "RLPDConfig",
    "BCPretrain",
    "ResidualActor",
    "IKExpert",
    "ik_step",
]
