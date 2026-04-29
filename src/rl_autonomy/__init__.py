"""rl_autonomy — RL pipeline for keyboard typing on the Rover2026 arm.

After Phase 1 of the rewrite this package will export:
    - envs.KeyboardEnv         (sim env)
    - algos.RLPDSAC            (training algorithm)
    - configs.controller_jp    (action-space controller config)
    - tools.*                  (one-off scripts: env diagnostics, demo recorder, ...)

Right now (Phase 0) the package only exposes the legacy keyboard_env
under its old top-level path so existing scripts that still import
`from keyboard_env import ...` keep working until Phase 1 rewrites the env.
"""

__version__ = "0.1.0.dev0"
