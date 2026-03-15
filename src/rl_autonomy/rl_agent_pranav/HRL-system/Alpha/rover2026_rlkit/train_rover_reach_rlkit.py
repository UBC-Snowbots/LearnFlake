from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from keypad_env import KeypadReachEnv, KeypadReachEnvConfig
from reach_env import Rover2026ReachEnv, RoverReachEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rover2026 reach policy with RLKit SAC")
    parser.add_argument("--env-kind", type=str, default="reach", choices=["reach", "keypad"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--expl-horizon", type=int, default=200)
    parser.add_argument("--eval-horizon", type=int, default=200)
    parser.add_argument("--expl-episodes-per-loop", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--trains-per-loop", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-before-training", type=int, default=1000)
    parser.add_argument("--replay-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--qf-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--soft-target-tau", type=float, default=5e-3)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=0.85)
    parser.add_argument("--max-delta-m", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--video-freq", type=int, default=20)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--run-name", type=str, default="rover2026_reach_rlkit")
    parser.add_argument("--init-policy-path", type=str, default="", help="Optional checkpoint to warm-start the policy from")
    parser.add_argument("--sample-key-targets", action="store_true", help="When using keypad env, sample goal positions from the keypad layout")
    parser.add_argument("--fixed-target-key", type=str, default="", help="When using keypad env, train/eval on a single key id")
    parser.add_argument("--key-target-noise-xy-m", type=float, default=0.0)
    parser.add_argument("--keypad-objective", type=str, default="hover", choices=["hover", "press"])
    parser.add_argument("--bc-demo-episodes", type=int, default=0, help="Collect scripted demo episodes and BC-pretrain the residual policy before SAC")
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--live-progress", action="store_true", help="Print compact live epoch progress")
    return parser.parse_args()


def evaluate_policy(policy, env: NormalizedBoxEnv, episodes: int, horizon: int) -> Dict[str, float]:
    returns = []
    success = []
    final_distance = []
    final_standoff_error = []

    for _ in range(episodes):
        path = rollout(env, policy, max_path_length=horizon, render=False)
        returns.append(float(np.sum(path["rewards"])))

        env_infos = path.get("env_infos", [])
        if env_infos:
            success.append(max(float(info.get("is_success", 0.0)) for info in env_infos))
            final_distance.append(float(env_infos[-1].get("distance_to_goal", np.nan)))
            final_standoff_error.append(float(env_infos[-1].get("standoff_error", np.nan)))

    return {
        "avg_return": float(np.mean(returns)) if returns else float("nan"),
        "success_rate": float(np.mean(success)) if success else 0.0,
        "avg_final_distance": float(np.nanmean(final_distance)) if final_distance else float("nan"),
        "avg_standoff_error": float(np.nanmean(final_standoff_error)) if final_standoff_error else float("nan"),
    }


def record_policy_video(policy, env: NormalizedBoxEnv, out_path: Path, horizon: int, fps: int = 20) -> bool:
    frames = []
    obs = env.reset()
    policy.reset()

    for _ in range(horizon):
        frame = env.wrapped_env.render(mode="rgb_array")
        frames.append(frame)
        action, _ = policy.get_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break

    if not frames:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return True


def save_policy_checkpoint(
    policy: TanhGaussianPolicy,
    path: Path,
    obs_dim: int,
    action_dim: int,
    hidden_sizes,
    env_config: RoverReachEnvConfig,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "policy_state_dict": policy.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_sizes": list(hidden_sizes),
        "env_config": asdict(env_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    torch.save(payload, path)


def maybe_warm_start_policy(policy: TanhGaussianPolicy, payload: Dict[str, object]) -> None:
    incoming = payload["policy_state_dict"]
    current = policy.state_dict()
    merged = dict(current)

    for name, tensor in incoming.items():
        if name not in current:
            continue
        if current[name].shape == tensor.shape:
            merged[name] = tensor
            continue
        # Allow warm-starting the input layer when observation size changes by
        # copying the shared prefix and leaving new features randomly initialized.
        if name.endswith("weight") and current[name].ndim == 2 and tensor.ndim == 2:
            out_ok = current[name].shape[0] == tensor.shape[0]
            in_ok = current[name].shape[1] >= tensor.shape[1]
            if out_ok and in_ok:
                blended = current[name].clone()
                blended[:, : tensor.shape[1]] = tensor
                merged[name] = blended
                continue

    policy.load_state_dict(merged)


def collect_scripted_hover_observations(env: NormalizedBoxEnv, episodes: int, horizon: int) -> np.ndarray:
    obs_samples = []
    if episodes <= 0:
        return np.empty((0, env.observation_space.low.size), dtype=np.float32)

    zero_action = np.zeros(env.action_space.low.shape, dtype=np.float32)
    for _ in range(episodes):
        obs = env.reset()
        obs_samples.append(np.asarray(obs, dtype=np.float32))
        for _ in range(horizon):
            obs, _, done, info = env.step(zero_action)
            obs_samples.append(np.asarray(obs, dtype=np.float32))
            if done:
                break

    return np.asarray(obs_samples, dtype=np.float32)


def behavior_clone_zero_residual(
    policy: TanhGaussianPolicy,
    observations: np.ndarray,
    epochs: int,
    batch_size: int,
    action_dim: int,
) -> float:
    if epochs <= 0 or observations.size == 0:
        return float("nan")

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=ptu.device)
    target = torch.zeros((obs_tensor.shape[0], action_dim), dtype=torch.float32, device=ptu.device)
    last_loss = float("nan")
    policy.train()

    for _ in range(epochs):
        permutation = torch.randperm(obs_tensor.shape[0], device=ptu.device)
        for start in range(0, obs_tensor.shape[0], batch_size):
            idx = permutation[start:start + batch_size]
            batch_obs = obs_tensor[idx]
            batch_target = target[idx]
            _, mean, _, _, _, _, _, _ = policy(batch_obs, deterministic=True)
            pred = torch.tanh(mean)
            loss = torch.nn.functional.mse_loss(pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    return last_loss


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    log_dir = (base_dir / args.log_dir).resolve()
    ckpt_dir = (base_dir / args.checkpoint_dir).resolve()
    video_dir = (base_dir / args.video_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    ptu.set_gpu_mode(device == "cuda")

    env_config_cls = RoverReachEnvConfig if args.env_kind == "reach" else KeypadReachEnvConfig
    env_cls = Rover2026ReachEnv if args.env_kind == "reach" else KeypadReachEnv

    common_cfg = dict(
        render=False,
        offscreen_render=False,
    )
    if args.max_delta_m is not None:
        common_cfg["max_delta_m"] = args.max_delta_m
    if args.env_kind == "keypad":
        common_cfg.update(
            sample_key_targets=bool(args.sample_key_targets),
            fixed_target_key=(args.fixed_target_key or None),
            key_target_noise_xy_m=args.key_target_noise_xy_m,
            objective_mode=args.keypad_objective,
        )
    expl_cfg = env_config_cls(
        horizon=args.expl_horizon,
        seed=args.seed,
        **common_cfg,
    )
    eval_cfg = env_config_cls(
        horizon=args.eval_horizon,
        seed=args.seed + 1,
        **common_cfg,
    )

    expl_env = NormalizedBoxEnv(env_cls(expl_cfg), reward_scale=args.reward_scale)
    eval_env = NormalizedBoxEnv(env_cls(eval_cfg), reward_scale=1.0)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    hidden_sizes = [256, 256]

    qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    target_qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)
    target_qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=hidden_sizes)

    policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)
    eval_policy = MakeDeterministic(policy)

    if args.init_policy_path:
        init_path = Path(args.init_policy_path)
        if not init_path.is_absolute():
            init_path = (base_dir / init_path).resolve()
        payload = torch.load(init_path, map_location="cpu")
        maybe_warm_start_policy(policy, payload)
        print(f"Warm-started policy from {init_path}")

    policy.to(ptu.device)

    if args.bc_demo_episodes > 0 and args.bc_epochs > 0:
        bc_obs = collect_scripted_hover_observations(expl_env, args.bc_demo_episodes, args.expl_horizon)
        bc_loss = behavior_clone_zero_residual(
            policy=policy,
            observations=bc_obs,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            action_dim=action_dim,
        )
        print(
            f"BC pretrain complete: demos={args.bc_demo_episodes} samples={bc_obs.shape[0]} "
            f"epochs={args.bc_epochs} final_loss={bc_loss:.6f}"
        )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        discount=args.gamma,
        soft_target_tau=args.soft_target_tau,
        target_update_period=1,
        policy_lr=args.policy_lr,
        qf_lr=args.qf_lr,
        reward_scale=args.reward_scale,
        use_automatic_entropy_tuning=True,
    )

    replay_buffer = EnvReplayBuffer(args.replay_buffer_size, expl_env)
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    expl_path_collector = MdpPathCollector(expl_env, policy)

    variant = {
        "algorithm": "SAC",
        "env_kind": args.env_kind,
        "seed": args.seed,
        "qf_hidden_sizes": hidden_sizes,
        "policy_hidden_sizes": hidden_sizes,
        "expl_cfg": asdict(expl_cfg),
        "eval_cfg": asdict(eval_cfg),
    }
    setup_logger(
        args.run_name,
        variant=variant,
        base_log_dir=str(log_dir),
        snapshot_mode="none",
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_size=args.batch_size,
        max_path_length=args.expl_horizon,
        num_epochs=args.num_epochs,
        num_eval_steps_per_epoch=args.eval_horizon * args.eval_episodes,
        num_expl_steps_per_train_loop=args.expl_horizon * args.expl_episodes_per_loop,
        num_trains_per_train_loop=args.trains_per_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=args.steps_before_training,
    )

    best_success = -np.inf
    best_path = ckpt_dir / "best_policy_rover2026_reach.pth"
    final_path = ckpt_dir / "final_policy_rover2026_reach.pth"

    start_time = time.time()
    prev_error = None

    def _progress_bar(done: int, total: int, width: int = 24) -> str:
        ratio = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
        filled = int(width * ratio)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {ratio * 100:5.1f}%"

    def post_epoch_hook(algo: TorchBatchRLAlgorithm, epoch: int):
        nonlocal best_success, prev_error

        metrics = evaluate_policy(eval_policy, eval_env, episodes=args.eval_episodes, horizon=args.eval_horizon)
        if args.live_progress:
            bar = _progress_bar(epoch + 1, args.num_epochs)
            prev_txt = "n/a" if prev_error is None else f"{prev_error:.4f}"
            print(
                f"{bar} epoch={epoch + 1}/{args.num_epochs} "
                f"ret={metrics['avg_return']:.3f} success={metrics['success_rate']:.3f} "
                f"standoff_err={metrics['avg_standoff_error']:.4f} prev_err={prev_txt}"
            )
        else:
            print(
                f"[epoch={epoch:04d}] avg_return={metrics['avg_return']:.3f} "
                f"success={metrics['success_rate']:.3f} "
                f"standoff_err={metrics['avg_standoff_error']:.4f} "
                f"final_dist={metrics['avg_final_distance']:.4f} "
                f"prev_standoff_err={'n/a' if prev_error is None else f'{prev_error:.4f}'}"
            )

        if metrics["success_rate"] > best_success:
            best_success = metrics["success_rate"]
            save_policy_checkpoint(
                policy=policy,
                path=best_path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                env_config=eval_cfg,
                epoch=epoch,
                metrics=metrics,
            )
            print(f"  saved new best policy -> {best_path}")

        if (epoch + 1) % args.checkpoint_freq == 0:
            periodic_path = ckpt_dir / f"policy_epoch_{epoch + 1:04d}.pth"
            save_policy_checkpoint(
                policy=policy,
                path=periodic_path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                env_config=eval_cfg,
                epoch=epoch,
                metrics=metrics,
            )

        if args.video_freq > 0 and ((epoch + 1) % args.video_freq == 0):
            video_path = video_dir / f"rollout_epoch_{epoch + 1:04d}.mp4"
            try:
                ok = record_policy_video(eval_policy, eval_env, video_path, horizon=args.eval_horizon)
                if ok:
                    print(f"  wrote progress video -> {video_path}")
            except Exception as exc:
                print(f"  video export skipped at epoch {epoch + 1}: {exc}")

        if metrics["success_rate"] >= args.success_threshold:
            print(
                f"Early stop triggered at epoch {epoch + 1} "
                f"(success_rate={metrics['success_rate']:.3f} >= {args.success_threshold:.3f})"
            )
            raise StopIteration

        prev_error = metrics["avg_standoff_error"]

    algorithm.post_epoch_funcs.append(post_epoch_hook)

    try:
        algorithm.to(ptu.device)
        algorithm.train()
    except StopIteration:
        pass
    finally:
        final_metrics = evaluate_policy(eval_policy, eval_env, episodes=args.eval_episodes, horizon=args.eval_horizon)
        save_policy_checkpoint(
            policy=policy,
            path=final_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            env_config=eval_cfg,
            epoch=args.num_epochs,
            metrics=final_metrics,
        )

        duration = time.time() - start_time
        print("\nTraining finished")
        print(f"  device: {device}")
        print(f"  duration_sec: {duration:.1f}")
        print(f"  final_avg_return: {final_metrics['avg_return']:.3f}")
        print(f"  final_success_rate: {final_metrics['success_rate']:.3f}")
        print(f"  final_avg_standoff_error: {final_metrics['avg_standoff_error']:.4f}")
        print(f"  final_avg_distance: {final_metrics['avg_final_distance']:.4f}")
        print(f"  best_policy: {best_path}")
        print(f"  final_policy: {final_path}")

        expl_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
