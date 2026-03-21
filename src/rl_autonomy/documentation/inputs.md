# Parameters

**venv**: `stable_baselines3.common.vec_env.VecEnv`
**expert_policy**: `stable_baselines3.common.policies.BasePolicy`
**scratch_dir**: `str` or `pathlib.Path`
**rng**: `np.random.Generator`
**bc_trainer**: `imitation.algorithms.bc.BC`
**beta_schedule**: `BetaSchedule` (optional)
**expert_trajs**: `Sequence[imitation.data.types.Trajectory]` (optional)