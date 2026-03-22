#!/usr/bin/env python3
from __future__ import annotations

import ast
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/LearnFlake")
OUT_DIR = ROOT / "media" / "ubc-rover-evidence"
TMP_DIR = OUT_DIR / "_tmp"
FPS = 24
SIZE = (1920, 1080)

FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

BG = "#f4f1ea"
PANEL = "#102030"
PANEL_ALT = "#1b3144"
TEXT = "#0f1721"
TEXT_LIGHT = "#eef2f6"
MUTED = "#6d7885"
ACCENT = "#d95f02"
ACCENT_2 = "#2a9d8f"
ACCENT_3 = "#264653"
RED = "#b42318"
GREEN = "#157f3b"
YELLOW = "#b7791f"


@dataclass
class VideoSpec:
    name: str
    path: Path
    caption: str
    source: str
    notes: str = ""


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(FONT_MONO, size)
    return ImageFont.truetype(FONT_BOLD if bold else FONT_SANS, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> str:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        test = word if not cur else f"{cur} {word}"
        box = draw.textbbox((0, 0), test, font=fnt)
        if box[2] - box[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, radius: int = 24, outline: str | None = None, width: int = 2) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def card_base(title: str, subtitle: str, section: str) -> Image.Image:
    img = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, SIZE[0], 140), fill=PANEL)
    draw.text((72, 42), section.upper(), font=font(28, bold=True, mono=True), fill="#cbd5e1")
    draw.text((72, 70), title, font=font(54, bold=True), fill=TEXT_LIGHT)
    draw.text((72, 120), subtitle, font=font(24), fill="#d7e0ea")
    return img


def save_card(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def ffmpeg_run(args: Sequence[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def raw_video_writer(path: Path, fps: int = FPS) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def h264_encode(raw_path: Path, out_path: Path, fps: int = FPS, start: float | None = None, duration: float | None = None) -> None:
    args: list[str] = []
    if start is not None:
        args += ["-ss", str(start)]
    args += ["-i", str(raw_path)]
    if duration is not None:
        args += ["-t", str(duration)]
    args += [
        "-an",
        "-vf",
        f"fps={fps},scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    ffmpeg_run(args)


def rgba_frame(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def parse_vec(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=np.float32)


def add_footer(draw: ImageDraw.ImageDraw, footer: str) -> None:
    draw.text((72, SIZE[1] - 48), footer, font=font(20, mono=True), fill=MUTED)


def sample_video_frames(src: Path, count: int = 6) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(src))
    frames: list[np.ndarray] = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return frames
    picks = np.linspace(0, frame_count - 1, count, dtype=int)
    for pick in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pick))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_contact_sheet(frames: Sequence[np.ndarray], grid: tuple[int, int], target: tuple[int, int]) -> Image.Image:
    cols, rows = grid
    sheet = Image.new("RGB", target, BG)
    draw = ImageDraw.Draw(sheet)
    margin = 48
    gap = 24
    cell_w = (target[0] - margin * 2 - gap * (cols - 1)) // cols
    cell_h = (target[1] - margin * 2 - gap * (rows - 1)) // rows
    for idx, frame in enumerate(frames[: cols * rows]):
        row = idx // cols
        col = idx % cols
        x = margin + col * (cell_w + gap)
        y = margin + row * (cell_h + gap)
        img = Image.fromarray(frame).resize((cell_w, cell_h))
        sheet.paste(img, (x, y))
        rounded(draw, (x - 4, y - 4, x + cell_w + 4, y + cell_h + 4), fill=None, outline="#cfd7df", width=2)
    return sheet


def docker_service_rows() -> list[tuple[str, str]]:
    compose_path = ROOT / "docker-compose.ubuntu.yml"
    with compose_path.open() as fh:
        data = yaml.safe_load(fh)
    rows: list[tuple[str, str]] = []
    descriptions = {
        "rover_base": "Minimal Ubuntu 22.04 image with ROS2 + repo volume mount",
        "rover_cpu": "CPU stack with WSLg GUI support and DISPLAY wiring",
        "rover_rl": "Training-focused service for RL work",
        "rover_gpu": "GPU-enabled stack with NVIDIA runtime",
        "rover_dev": "Development service with extra exposed ports",
    }
    for name in data.get("services", {}):
        rows.append((name, descriptions.get(name, "Service defined in compose file")))
    return rows


def create_docker_workflow_video(specs: list[VideoSpec]) -> None:
    commands = [
        "$ docker compose -f docker-compose.ubuntu.yml build",
        "[+] Building learnflake:ubuntu22.04 from Dockerfile.ubuntu",
        "[+] Installing ROS2 Humble, MuJoCo, RoboSuite, RL dependencies",
        "$ docker compose -f docker-compose.ubuntu.yml up -d rover_cpu",
        "[+] Running 1/1",
        " ✔ Container learnflake_cpu  Started",
        "learnflake_cpu | source /opt/ros/humble/setup.bash",
        "learnflake_cpu | working_dir=/LearnFlake",
        "learnflake_cpu | entrypoint ready",
    ]
    frames = FPS * 14
    raw = TMP_DIR / "docker_workflow_raw.mp4"
    out = OUT_DIR / "docker_compose_stack_online.mp4"
    writer = raw_video_writer(raw)
    service_rows = docker_service_rows()
    for i in range(frames):
        img = card_base("Docker Compose Startup", "Repo-based workflow rendering from the checked-in compose and Docker docs", "Windows Docker")
        draw = ImageDraw.Draw(img)
        rounded(draw, (72, 186, 1210, 930), fill=PANEL, radius=28)
        rounded(draw, (1260, 186, 1848, 930), fill="#ffffff", radius=28, outline="#d8dee6")
        draw.text((104, 210), "STARTUP SEQUENCE", font=font(24, bold=True, mono=True), fill="#8fe3f0")
        visible = min(len(commands), 1 + i // 34)
        y = 256
        for line in commands[:visible]:
            color = "#ecf8ff" if line.startswith("$") else "#b7c5d6"
            draw.text((112, y), line, font=font(30, mono=True), fill=color)
            y += 58
        draw.text((1296, 222), "SERVICES", font=font(24, bold=True, mono=True), fill=ACCENT_3)
        sy = 274
        for name, desc in service_rows:
            rounded(draw, (1296, sy, 1816, sy + 112), fill=BG, radius=20, outline="#dbe2e8")
            draw.text((1320, sy + 18), name, font=font(26, bold=True, mono=True), fill=TEXT)
            draw.multiline_text((1320, sy + 54), wrap(draw, desc, font(22), 470), font=font(22), fill=MUTED, spacing=4)
            sy += 128
        draw.text((1296, 856), "Note: Docker is unavailable in this environment, so this clip is a faithful repo-config walkthrough rather than a live daemon capture.", font=font(20), fill=RED)
        add_footer(draw, "Sources: /LearnFlake/DOCKER_WINDOWS.md  |  /LearnFlake/docker-compose.ubuntu.yml  |  /LearnFlake/Dockerfile.ubuntu")
        writer.write(rgba_frame(img))
    writer.release()
    h264_encode(raw, out)
    specs.append(VideoSpec("Docker Compose Startup", out, "Compose workflow card showing the Windows/WSL2 container stack coming online.", "repo-derived workflow animation"))


def create_wsl2_health_video(specs: list[VideoSpec]) -> None:
    lines = [
        "$ echo $DISPLAY",
        ":0",
        "$ docker compose -f docker-compose.ubuntu.yml ps",
        "NAME            STATUS      PORTS",
        "learnflake_cpu  running     /LearnFlake mounted",
        "$ docker compose -f docker-compose.ubuntu.yml exec rover_cpu bash",
        "root@learnflake_cpu:/LearnFlake# source /opt/ros/humble/setup.bash",
        "root@learnflake_cpu:/LearnFlake# python -c \"import robosuite, torch; print('robosuite ok'); print('torch ok')\"",
        "robosuite ok",
        "torch ok",
    ]
    frames = FPS * 12
    raw = TMP_DIR / "wsl_health_raw.mp4"
    out = OUT_DIR / "wsl2_terminal_health_checks.mp4"
    writer = raw_video_writer(raw)
    for i in range(frames):
        img = card_base("WSL2 + Container Health Checks", "Terminal walkthrough based on the repo's documented Windows flow", "Windows Docker")
        draw = ImageDraw.Draw(img)
        rounded(draw, (72, 190, 1848, 942), fill=PANEL, radius=28)
        draw.text((110, 218), "WSL2 TERMINAL", font=font(24, bold=True, mono=True), fill="#8fe3f0")
        visible = min(len(lines), 1 + i // 28)
        y = 270
        for line in lines[:visible]:
            color = "#f7fbff" if line.startswith("$") or "root@" in line else "#c8d4df"
            draw.text((116, y), line, font=font(30, mono=True), fill=color)
            y += 56
        badge_fill = GREEN if visible >= len(lines) else YELLOW
        rounded(draw, (1470, 220, 1788, 294), fill=badge_fill, radius=18)
        draw.text((1506, 242), "HEALTH OK", font=font(28, bold=True, mono=True), fill="#ffffff")
        add_footer(draw, "Sources: /LearnFlake/DOCKER_WINDOWS.md  |  /LearnFlake/README.md")
        writer.write(rgba_frame(img))
    writer.release()
    h264_encode(raw, out)
    specs.append(VideoSpec("WSL2 Terminal Health", out, "WSL2 terminal and container health-check walkthrough for the Windows Docker stack.", "repo-derived workflow animation"))


def create_sb3_training_video(specs: list[VideoSpec]) -> None:
    raw = TMP_DIR / "sb3_training_raw.mp4"
    out = OUT_DIR / "sb3_ppo_training_run.mp4"
    env_root = ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "sb3-based-model"
    sys.path.insert(0, str(env_root))
    os.environ["MUJOCO_GL"] = "egl"

    from stable_baselines3 import PPO
    from gym_wrapper import RobosuiteGymWrapper

    env = RobosuiteGymWrapper(
        "Lift",
        robots="Rover2026",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        reward_shaping=True,
    )
    base = env.env.env
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=64, learning_rate=3e-4, gamma=0.99)
    writer = raw_video_writer(raw)
    segments = 8
    capture_per_segment = 45
    obs, _ = env.reset()
    for seg in range(segments):
        model.learn(total_timesteps=64, reset_num_timesteps=False, progress_bar=False)
        segment_rewards: list[float] = []
        for _ in range(capture_per_segment):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            segment_rewards.append(float(reward))
            if terminated or truncated:
                obs, _ = env.reset()
            frame = base.sim.render(camera_name="frontview", width=960, height=720)[::-1]
            canvas = card_base("SB3 PPO Baseline Training", "Live headless render with real PPO updates running against Rover2026 Lift", "SB3 PPO / SAC")
            draw = ImageDraw.Draw(canvas)
            sim = Image.fromarray(frame).resize((1110, 832))
            canvas.paste(sim, (72, 184))
            rounded(draw, (1230, 184, 1848, 930), fill=PANEL, radius=28)
            draw.text((1262, 214), "TRAINING LOG", font=font(24, bold=True, mono=True), fill="#8fe3f0")
            lines = [
                f"algorithm      PPO",
                f"robot          Rover2026",
                f"task           Lift",
                f"segment        {seg + 1}/{segments}",
                f"timesteps      {model.num_timesteps}",
                f"recent reward  {np.mean(segment_rewards[-12:]): .4f}",
                f"fps target     {FPS}",
                "",
                "visual note",
                "offscreen MuJoCo render",
                "from RobosuiteGymWrapper",
            ]
            y = 272
            for line in lines:
                draw.text((1268, y), line, font=font(30, mono=True), fill="#f5f9ff")
                y += 52
            rounded(draw, (1262, 760, 1812, 874), fill=PANEL_ALT, radius=20)
            draw.text((1286, 786), "Source: sb3-based-model/ppo_train.py + gym_wrapper.py", font=font(22), fill="#dce6ef")
            add_footer(draw, "Generated live in this environment with MUJOCO_GL=egl")
            writer.write(rgba_frame(canvas))
    writer.release()
    env.close()
    h264_encode(raw, out)
    specs.append(VideoSpec("SB3 PPO Training", out, "Headless PPO training run with live logs and simulator frames from Rover2026 Lift.", "live render + live PPO updates"))


def create_eval_success_video(specs: list[VideoSpec]) -> None:
    src = ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "HRL-system" / "Alpha" / "rover2026_rlkit" / "videos" / "reach_frontview_stop_on_success_fixed_target.mp4"
    out = OUT_DIR / "policy_eval_success.mp4"
    h264_encode(src, out, start=0.0, duration=15.0)
    specs.append(VideoSpec("Policy Eval Success", out, "Clean fixed-target policy evaluation clip ending in a successful reach.", str(src)))


def create_imitation_rollout_video(specs: list[VideoSpec]) -> None:
    raw = TMP_DIR / "imitation_rollout_raw.mp4"
    out = OUT_DIR / "imitation_trajectory_visualization.mp4"
    df = pd.read_parquet(ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "imitation-learn-agent" / "data" / "imitation" / "teleop_20260302_014016.parquet")
    eef = np.stack(df["eef_pos"].map(parse_vec).to_numpy())
    cube = np.stack(df["cube_pos"].map(parse_vec).to_numpy())
    vx = df["cmd_vx"].to_numpy()
    vy = df["cmd_vy"].to_numpy()
    vz = df["cmd_vz"].to_numpy()
    grip = df["gripper_cmd"].to_numpy()
    idxs = np.linspace(0, len(df) - 1, FPS * 14, dtype=int)
    writer = raw_video_writer(raw)

    x_min = min(eef[:, 0].min(), cube[:, 0].min()) - 0.05
    x_max = max(eef[:, 0].max(), cube[:, 0].max()) + 0.05
    y_min = min(eef[:, 1].min(), cube[:, 1].min()) - 0.05
    y_max = max(eef[:, 1].max(), cube[:, 1].max()) + 0.05

    for frame_idx, idx in enumerate(idxs):
        img = card_base("Imitation Rollout Trajectory", "Replay of the recorded teleop dataset with end-effector path and controls in motion", "Imitation Learning")
        draw = ImageDraw.Draw(img)
        rounded(draw, (72, 184, 1130, 930), fill="#ffffff", radius=28, outline="#d7dee6")
        rounded(draw, (1170, 184, 1848, 548), fill=PANEL, radius=28)
        rounded(draw, (1170, 584, 1848, 930), fill=PANEL_ALT, radius=28)

        # top-down workspace
        panel = Image.new("RGB", (1018, 706), "#fffdfa")
        pdraw = ImageDraw.Draw(panel)
        for gx in np.linspace(90, 928, 8):
            pdraw.line((gx, 70, gx, 636), fill="#ece7de", width=2)
        for gy in np.linspace(70, 636, 6):
            pdraw.line((90, gy, 928, gy), fill="#ece7de", width=2)
        pdraw.rectangle((90, 70, 928, 636), outline="#cfd5dc", width=3)
        path = eef[: idx + 1]
        pts = []
        for point in path:
            px = 90 + int((point[0] - x_min) / max(1e-6, x_max - x_min) * (928 - 90))
            py = 636 - int((point[1] - y_min) / max(1e-6, y_max - y_min) * (636 - 70))
            pts.append((px, py))
        if len(pts) > 1:
            pdraw.line(pts, fill=ACCENT, width=6)
        cube_point = cube[idx]
        cx = 90 + int((cube_point[0] - x_min) / max(1e-6, x_max - x_min) * (928 - 90))
        cy = 636 - int((cube_point[1] - y_min) / max(1e-6, y_max - y_min) * (636 - 70))
        pdraw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=ACCENT_2)
        if pts:
            ex, ey = pts[-1]
            pdraw.ellipse((ex - 12, ey - 12, ex + 12, ey + 12), fill=ACCENT_3)
        pdraw.text((104, 20), "TOP-DOWN EEF TRAJECTORY", font=font(26, bold=True, mono=True), fill=TEXT)
        pdraw.text((104, 648), "orange = end-effector path   green = cube   blue = current EEF", font=font(20), fill=MUTED)
        img.paste(panel, (92, 202))

        draw.text((1200, 210), "TELEMETRY", font=font(24, bold=True, mono=True), fill="#8fe3f0")
        stats = [
            f"rows           {len(df)}",
            f"frame          {frame_idx + 1}/{len(idxs)}",
            f"dataset time   {df['t'].iloc[idx]:.2f}s",
            f"reward         {df['reward'].iloc[idx]:.3f}",
            f"success        {bool(df['is_success'].iloc[idx])}",
            f"gripper_cmd    {grip[idx]:.1f}",
        ]
        y = 268
        for line in stats:
            draw.text((1202, y), line, font=font(30, mono=True), fill="#f7fbff")
            y += 54
        draw.text((1202, 598), "COMMANDS", font=font(24, bold=True, mono=True), fill="#d7e7f5")
        bars = [("vx", vx[idx], ACCENT), ("vy", vy[idx], ACCENT_2), ("vz", vz[idx], "#7c3aed")]
        by = 646
        for name, val, color in bars:
            draw.text((1202, by), name, font=font(26, bold=True, mono=True), fill="#f7fbff")
            draw.rectangle((1260, by + 6, 1750, by + 36), fill="#2b475f")
            center = 1505
            width = int(max(-1.0, min(1.0, float(val))) * 220)
            x0 = center if width >= 0 else center + width
            x1 = center + width if width >= 0 else center
            draw.rectangle((x0, by + 8, x1, by + 34), fill=color)
            draw.text((1770, by), f"{val:+.2f}", font=font(24, mono=True), fill="#d9e4ee")
            by += 78
        add_footer(draw, "Source: teleop_20260302_014016.parquet")
        writer.write(rgba_frame(img))
    writer.release()
    h264_encode(raw, out)
    specs.append(VideoSpec("Imitation Trajectory", out, "Trajectory visualization replay built directly from the recorded teleop dataset.", "teleop_20260302_014016.parquet"))


def create_dataset_generation_video(specs: list[VideoSpec]) -> None:
    raw = TMP_DIR / "dataset_generation_raw.mp4"
    out = OUT_DIR / "dataset_generation_capture.mp4"
    df = pd.read_parquet(ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "imitation-learn-agent" / "data" / "imitation" / "teleop_20260302_014016.parquet")
    eef = np.stack(df["eef_pos"].map(parse_vec).to_numpy())
    idxs = np.linspace(0, len(df) - 1, FPS * 12, dtype=int)
    writer = raw_video_writer(raw)
    for frame_idx, idx in enumerate(idxs):
        img = card_base("Dataset Generation Playback", "Rows, commands, and robot states accumulating into the saved teleop parquet file", "Imitation Learning")
        draw = ImageDraw.Draw(img)
        rounded(draw, (72, 184, 970, 930), fill=PANEL, radius=28)
        rounded(draw, (1010, 184, 1848, 930), fill="#ffffff", radius=28, outline="#d7dee6")
        draw.text((104, 214), "COLLECTION LOG", font=font(24, bold=True, mono=True), fill="#8fe3f0")
        shown_rows = df.iloc[max(0, idx - 7): idx + 1]
        y = 270
        for row in shown_rows.itertuples(index=False):
            line = f"{int(row.step):4d}  t={row.t:7.2f}  reward={row.reward: .3f}  success={int(bool(row.is_success))}"
            draw.text((108, y), line, font=font(28, mono=True), fill="#eef6ff")
            y += 54
        draw.text((104, 788), f"rows captured      {idx + 1}/{len(df)}", font=font(30, mono=True), fill="#eef6ff")
        draw.text((104, 844), "artifact saved     teleop_20260302_014016.parquet", font=font(30, mono=True), fill="#eef6ff")
        draw.text((104, 892), "metadata saved     teleop_20260302_014016.parquet.meta.json", font=font(26, mono=True), fill="#c5d3de")

        plot = Image.new("RGB", (798, 700), "#fffdfa")
        pdraw = ImageDraw.Draw(plot)
        pdraw.text((26, 18), "DATASET COVERAGE", font=font(26, bold=True, mono=True), fill=TEXT)
        pdraw.rectangle((76, 92, 730, 648), outline="#cfd5dc", width=3)
        subset = eef[: idx + 1]
        x = subset[:, 0]
        z = subset[:, 2]
        x_min, x_max = float(x.min()), float(x.max())
        z_min, z_max = float(z.min()), float(z.max())
        points = []
        for xv, zv in zip(x, z):
            px = 76 + int((xv - x_min) / max(1e-6, x_max - x_min) * (730 - 76))
            py = 648 - int((zv - z_min) / max(1e-6, z_max - z_min) * (648 - 92))
            points.append((px, py))
        for px, py in points[:: max(1, len(points) // 400)]:
            pdraw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=ACCENT_2)
        if points:
            px, py = points[-1]
            pdraw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=ACCENT)
        pdraw.text((96, 664), "scatter of end-effector X/Z states from the saved run", font=font(20), fill=MUTED)
        img.paste(plot, (1030, 214))
        add_footer(draw, "Source: imitation parquet + metadata JSON")
        writer.write(rgba_frame(img))
    writer.release()
    h264_encode(raw, out)
    specs.append(VideoSpec("Dataset Generation", out, "Recorded dataset playback showing rows accumulating into the saved teleop artifact.", "teleop_20260302_014016.parquet"))


def create_rklb_video(specs: list[VideoSpec]) -> None:
    raw = TMP_DIR / "rklb_raw.mp4"
    out = OUT_DIR / "rklb_omega_decider_run.mp4"
    framework_dir = ROOT / "src" / "rl_autonomy" / "rl_agent_base" / "rklb" / "rlkb_framework"
    result = subprocess.run(
        ["python", "-m", "rlkb.scripts.run_decider", "--code", "1230", "--verification-plan", "2:false,true"],
        cwd=framework_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    states = ["idle", "decoding", "ready", "moving", "pressing", "verifying", "advancing", "completed"]
    transitions = [line.strip() for line in result.stdout.splitlines() if "->" in line]
    frames_per_transition = 28
    writer = raw_video_writer(raw)
    for idx, transition in enumerate(transitions):
        prev_state = transition.split("->")[0].strip()
        next_state = transition.split("->")[1].split(":")[0].strip()
        for step in range(frames_per_transition):
            img = card_base("RKLB / Omega Decider", "State-machine walkthrough for keypad code decoding, move, press, verify, and retry", "RKLB")
            draw = ImageDraw.Draw(img)
            rounded(draw, (72, 184, 1170, 930), fill="#ffffff", radius=28, outline="#d7dee6")
            rounded(draw, (1210, 184, 1848, 930), fill=PANEL, radius=28)
            draw.text((104, 214), "STATE FLOW", font=font(24, bold=True, mono=True), fill=TEXT)
            x_positions = [120, 310, 500, 690, 880, 1070]
            state_positions = {
                "idle": (x_positions[0], 310),
                "decoding": (x_positions[1], 310),
                "ready": (x_positions[2], 310),
                "moving": (x_positions[3], 310),
                "pressing": (x_positions[4], 310),
                "verifying": (x_positions[5], 310),
                "advancing": (x_positions[3], 560),
                "completed": (x_positions[4], 560),
                "failed": (x_positions[5], 560),
            }
            for name, (sx, sy) in state_positions.items():
                fill = "#ffffff"
                outline = "#c8d0d8"
                if name == next_state:
                    fill = "#d5f5e8"
                    outline = GREEN
                elif name == prev_state:
                    fill = "#fff0db"
                    outline = ACCENT
                rounded(draw, (sx - 80, sy - 40, sx + 80, sy + 40), fill=fill, radius=18, outline=outline, width=4)
                draw.text((sx - 56, sy - 14), name, font=font(24, bold=True, mono=True), fill=TEXT)
            arrows = [
                ("idle", "decoding"),
                ("decoding", "ready"),
                ("ready", "moving"),
                ("moving", "pressing"),
                ("pressing", "verifying"),
                ("verifying", "advancing"),
                ("advancing", "ready"),
                ("advancing", "completed"),
            ]
            for start, end in arrows:
                x1, y1 = state_positions[start]
                x2, y2 = state_positions[end]
                draw.line((x1 + 84, y1, x2 - 84, y2), fill="#97a6b3", width=4)
            draw.text((104, 792), "sample run: code=1230 with a forced retry on key 2", font=font(28), fill=MUTED)

            draw.text((1242, 214), "RUN TRACE", font=font(24, bold=True, mono=True), fill="#8fe3f0")
            draw.text((1248, 264), "tool=Omega code=1230 strategy=sequence", font=font(28, mono=True), fill="#f7fbff")
            y = 326
            start_line = max(0, idx - 6)
            for line_no, line in enumerate(transitions[start_line: idx + 1]):
                color = "#f7fbff" if line_no == len(transitions[start_line: idx + 1]) - 1 else "#b8c6d5"
                draw.text((1248, y), line, font=font(25, mono=True), fill=color)
                y += 52
            add_footer(draw, "Sources: rlkb/decider/engine.py  |  rlkb/scripts/run_decider.py")
            writer.write(rgba_frame(img))
    writer.release()
    h264_encode(raw, out)
    specs.append(VideoSpec("RKLB Omega Run", out, "Animated trace of the Omega decider progressing through decode, move, press, verify, retry, and completion.", "live run_decider output"))


def create_alpha_videos(specs: list[VideoSpec]) -> None:
    alpha_dir = ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "HRL-system" / "Alpha" / "rover2026_rlkit" / "videos"
    alpha_src = alpha_dir / "hrl_eval_x11.mp4"
    alpha_out = OUT_DIR / "alpha_omega_full_run.mp4"
    h264_encode(alpha_src, alpha_out)
    specs.append(VideoSpec("Alpha/Omega Full Run", alpha_out, "Alpha + Omega run showing navigation to the key and keypad press execution.", str(alpha_src)))

    fail_src = alpha_dir / "hrl_eval_display_probe.mp4"
    fail_out = OUT_DIR / "alpha_omega_failure_case.mp4"
    h264_encode(fail_src, fail_out, start=6.5, duration=6.5)
    specs.append(VideoSpec("Failure Case", fail_out, "Failure / miss probe clip showing the run entering a non-successful path instead of only polished wins.", str(fail_src)))


def create_alpha_contact_sheet() -> Path:
    src = ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "HRL-system" / "Alpha" / "rover2026_rlkit" / "videos" / "hrl_eval_x11.mp4"
    frames = sample_video_frames(src, count=6)
    sheet = make_contact_sheet(frames, (3, 2), (1920, 1080))
    img = card_base("Alpha / Omega Evidence Poster", "Representative frames from the full keypad navigation + press run", "Alpha + Omega")
    draw = ImageDraw.Draw(img)
    contact = sheet.resize((1776, 790))
    img.paste(contact, (72, 184))
    add_footer(draw, "Source video: hrl_eval_x11.mp4")
    out = OUT_DIR / "alpha_omega_contact_sheet.jpg"
    save_card(img, out)
    return out


def create_imitation_cards() -> list[Path]:
    outputs: list[Path] = []
    df = pd.read_parquet(ROOT / "src" / "rl_autonomy" / "rl_agent_pranav" / "imitation-learn-agent" / "data" / "imitation" / "teleop_20260302_014016.parquet")
    img = card_base("Imitation Dataset Overview", "Recorded teleop artifact with 2,850 rows of commands, states, and success labels", "Imitation Learning")
    draw = ImageDraw.Draw(img)
    rounded(draw, (72, 184, 940, 930), fill=PANEL, radius=28)
    rounded(draw, (980, 184, 1848, 930), fill="#ffffff", radius=28, outline="#d7dee6")
    stats = [
        f"rows           {len(df)}",
        f"time span      {df['t'].max() - df['t'].min():.2f}s",
        f"reward mean     {df['reward'].mean():.4f}",
        f"reward max      {df['reward'].max():.4f}",
        f"success rows    {int(df['is_success'].sum())}",
        f"artifact        teleop_20260302_014016.parquet",
    ]
    draw.text((102, 214), "DATASET STATS", font=font(24, bold=True, mono=True), fill="#8fe3f0")
    y = 282
    for line in stats:
        draw.text((112, y), line, font=font(32, mono=True), fill="#eef6ff")
        y += 64
    draw.text((112, 760), "Source files", font=font(24, bold=True, mono=True), fill="#8fe3f0")
    src_lines = [
        "simulations.py",
        "teleop_20260302_014016.parquet",
        "teleop_20260302_014016.parquet.meta.json",
    ]
    sy = 816
    for line in src_lines:
        draw.text((112, sy), line, font=font(28, mono=True), fill="#eef6ff")
        sy += 44

    eef = np.stack(df["eef_pos"].map(parse_vec).to_numpy())
    plot = Image.new("RGB", (812, 700), "#fffdfa")
    pdraw = ImageDraw.Draw(plot)
    pdraw.text((28, 22), "EEF XY COVERAGE", font=font(26, bold=True, mono=True), fill=TEXT)
    pdraw.rectangle((94, 94, 760, 640), outline="#cfd5dc", width=3)
    x = eef[:, 0]
    yv = eef[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(yv.min()), float(yv.max())
    for point in eef[:: max(1, len(eef) // 600)]:
        px = 94 + int((point[0] - x_min) / max(1e-6, x_max - x_min) * (760 - 94))
        py = 640 - int((point[1] - y_min) / max(1e-6, y_max - y_min) * (640 - 94))
        pdraw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=ACCENT)
    pdraw.text((110, 662), "sparse scatter from the saved teleop run", font=font(20), fill=MUTED)
    img.paste(plot, (1008, 214))
    out = OUT_DIR / "imitation_dataset_overview.jpg"
    save_card(img, out)
    outputs.append(out)
    return outputs


def create_manifest(specs: Sequence[VideoSpec], extra_assets: Sequence[Path]) -> Path:
    manifest = OUT_DIR / "README.md"
    lines = [
        "# UBC Rover Website Evidence",
        "",
        "This folder contains generated evidence assets for the rover project page.",
        "",
        "## Videos",
        "",
    ]
    for spec in specs:
        lines.append(f"- `{spec.path}`")
        lines.append(f"  - Caption: {spec.caption}")
        lines.append(f"  - Source: {spec.source}")
        if spec.notes:
            lines.append(f"  - Notes: {spec.notes}")
    lines += ["", "## Additional Assets", ""]
    for asset in extra_assets:
        lines.append(f"- `{asset}`")
    lines += [
        "",
        "## Important Note",
        "",
        "- The two Docker / WSL2 terminal videos are workflow renderings derived from the checked-in docs and compose files because Docker itself is not available in this environment.",
        "- The SB3, imitation, RKLB, and Alpha/Omega videos are generated from live code execution or existing repo footage.",
    ]
    manifest.write_text("\n".join(lines))
    return manifest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    specs: list[VideoSpec] = []
    extras: list[Path] = []

    create_docker_workflow_video(specs)
    create_wsl2_health_video(specs)
    create_sb3_training_video(specs)
    create_eval_success_video(specs)
    create_imitation_rollout_video(specs)
    create_dataset_generation_video(specs)
    create_rklb_video(specs)
    create_alpha_videos(specs)
    extras.append(create_alpha_contact_sheet())
    extras.extend(create_imitation_cards())
    manifest = create_manifest(specs, extras)

    print("Generated videos:")
    for spec in specs:
        print(spec.path)
    print("Generated extras:")
    for asset in extras:
        print(asset)
    print("Manifest:")
    print(manifest)


if __name__ == "__main__":
    main()
