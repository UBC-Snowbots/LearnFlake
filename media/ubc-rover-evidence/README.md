# UBC Rover Website Evidence

This folder contains generated evidence assets for the rover project page.

## Current Recommendation

Use these now:
- `/LearnFlake/media/ubc-rover-evidence/policy_eval_success.mp4`
- `/LearnFlake/media/ubc-rover-evidence/imitation_trajectory_visualization.mp4`
- `/LearnFlake/media/ubc-rover-evidence/dataset_generation_capture.mp4`
- `/LearnFlake/media/ubc-rover-evidence/rklb_omega_decider_run.mp4`
- `/LearnFlake/media/ubc-rover-evidence/alpha_omega_failure_case.mp4`
- `/LearnFlake/media/ubc-rover-evidence/sb3_baseline_overview.jpg`
- `/LearnFlake/media/ubc-rover-evidence/imitation_dataset_overview.jpg`
- `/LearnFlake/media/ubc-rover-evidence/alpha_omega_contact_sheet.jpg`

Use with caution:
- `/LearnFlake/media/ubc-rover-evidence/docker_compose_stack_online.mp4`
- `/LearnFlake/media/ubc-rover-evidence/wsl2_terminal_health_checks.mp4`
- `/LearnFlake/media/ubc-rover-evidence/sb3_sac_eval_live_h264.mp4`

Do not present as strong success evidence:
- `/LearnFlake/media/ubc-rover-evidence/sb3_ppo_training_run.mp4`
- `/LearnFlake/media/ubc-rover-evidence/alpha_omega_full_run.mp4`

## Videos

- `/LearnFlake/media/ubc-rover-evidence/docker_compose_stack_online.mp4`
  - Caption: Compose workflow card showing the Windows/WSL2 container stack coming online.
  - Source: repo-derived workflow animation
- `/LearnFlake/media/ubc-rover-evidence/wsl2_terminal_health_checks.mp4`
  - Caption: WSL2 terminal and container health-check walkthrough for the Windows Docker stack.
  - Source: repo-derived workflow animation
- `/LearnFlake/media/ubc-rover-evidence/sb3_ppo_training_run.mp4`
  - Caption: Headless PPO training run with live logs and simulator frames from Rover2026 Lift.
  - Source: live render + live PPO updates
- `/LearnFlake/media/ubc-rover-evidence/policy_eval_success.mp4`
  - Caption: Clean fixed-target policy evaluation clip ending in a successful reach.
  - Source: /LearnFlake/src/rl_autonomy/rl_agent_pranav/HRL-system/Alpha/rover2026_rlkit/videos/reach_frontview_stop_on_success_fixed_target.mp4
- `/LearnFlake/media/ubc-rover-evidence/imitation_trajectory_visualization.mp4`
  - Caption: Trajectory visualization replay built directly from the recorded teleop dataset.
  - Source: teleop_20260302_014016.parquet
- `/LearnFlake/media/ubc-rover-evidence/dataset_generation_capture.mp4`
  - Caption: Recorded dataset playback showing rows accumulating into the saved teleop artifact.
  - Source: teleop_20260302_014016.parquet
- `/LearnFlake/media/ubc-rover-evidence/rklb_omega_decider_run.mp4`
  - Caption: Animated trace of the Omega decider progressing through decode, move, press, verify, retry, and completion.
  - Source: live run_decider output
- `/LearnFlake/media/ubc-rover-evidence/alpha_omega_full_run.mp4`
  - Caption: Alpha + Omega run showing navigation to the key and keypad press execution.
  - Source: /LearnFlake/src/rl_autonomy/rl_agent_pranav/HRL-system/Alpha/rover2026_rlkit/videos/hrl_eval_x11.mp4
- `/LearnFlake/media/ubc-rover-evidence/alpha_omega_failure_case.mp4`
  - Caption: Failure / miss probe clip showing the run entering a non-successful path instead of only polished wins.
  - Source: /LearnFlake/src/rl_autonomy/rl_agent_pranav/HRL-system/Alpha/rover2026_rlkit/videos/hrl_eval_display_probe.mp4

## Additional Assets

- `/LearnFlake/media/ubc-rover-evidence/alpha_omega_contact_sheet.jpg`
- `/LearnFlake/media/ubc-rover-evidence/imitation_dataset_overview.jpg`
- `/LearnFlake/media/ubc-rover-evidence/sb3_baseline_overview.jpg`
- `/LearnFlake/media/ubc-rover-evidence/sb3_sac_rollout_rewards.png`
- `/LearnFlake/media/ubc-rover-evidence/sb3_sac_eval_live_h264.mp4`

## Important Note

- The two Docker / WSL2 terminal videos are workflow renderings derived from the checked-in docs and compose files because Docker itself is not available in this environment.
- The SB3, imitation, RKLB, and Alpha/Omega videos are generated from live code execution or existing repo footage.
- The saved SB3 checkpoints are mixed quality. `sac_Rover2026_V1_model.zip` runs, but the visual result is still not strong enough to use as polished success evidence.
- A true live Alpha/Omega rerun currently fails in this environment with a policy / observation-dimension mismatch (`18` expected vs `21` actual), so the existing `alpha_omega_full_run.mp4` should not be treated as a clean verified success clip.
- For real Docker / WSL2 capture on the host machine, use:
  - `/LearnFlake/DOCKER_EVIDENCE_CAPTURE.md`
  - `/LearnFlake/tools/capture_docker_evidence.sh`
