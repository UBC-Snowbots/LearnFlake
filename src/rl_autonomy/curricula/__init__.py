"""rl_autonomy.curricula — task-distribution schedulers wrapping KeyboardEnv.

Public API:
    KeyPhaseCurriculum   -- TRACKER §7.2: phase A (central keys) → B → C (full 87)
                            advance based on rolling per-phase success rate.
    StateReplayCurriculum -- TRACKER §7.1: DemoStart-style state replay.
                            v1 stub since demos are skipped per user direction.
"""
from .key_phase_curriculum import KeyPhaseCurriculum
from .state_replay_curriculum import StateReplayCurriculum

__all__ = ["KeyPhaseCurriculum", "StateReplayCurriculum"]
