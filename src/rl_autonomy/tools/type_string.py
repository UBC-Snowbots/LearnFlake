"""Type a string in simulation — the arm approaches and presses each key.

Feeds a string of characters to the solved Approach→Strike pipeline (residual-on-IK
Approach + open-loop solenoid Strike, TRACKER §37–§39) and drives the MuJoCo arm
to type it out, key by key, in one continuous motion. Renders an annotated MP4
(or a live viewer with --interactive, needs X11), with an overlay showing the
target key and the text typed so far.

Each character: switch the residual Approach to that key's column → drive until
within 4 mm (the residual policy) → switch to Strike in-process → extend the
solenoid until contact-hold → retract → next character.

Run inside rover_gpu:
    python3 -m rl_autonomy.tools.type_string \\
        --text "hello world" \\
        --out /LearnFlake/typing.mp4

    # then copy it out of the container:
    docker cp rover_gpu:/LearnFlake/typing.mp4 .

Defaults use the committed M4 model (models/approach_v16b_residual.pt) at the
tuned keyboard position (-0.10,-0.10) with tube 0.15 — these MUST match the
checkpoint (see README / TRACKER §39).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# character -> key-name mapping
# ---------------------------------------------------------------------------

_PUNCT = {
    " ": "space", "\n": "enter", "\t": "tab",
    "-": "minus", "=": "equal", "[": "lbracket", "]": "rbracket",
    "\\": "backslash", ";": "semicolon", "'": "quote", ",": "comma",
    ".": "period", "/": "slash", "`": "grave",
}


def char_to_key(ch: str) -> str | None:
    """Map a typeable character to a key name, or None if unmappable.

    Uppercase letters are lowered (shift is not modelled). Returns None for
    characters with no single-key equivalent (they are skipped with a warning).
    """
    if ch.isalpha():
        return ch.lower()
    if ch.isdigit():
        return ch
    return _PUNCT.get(ch)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

def _annotate(frame: np.ndarray, typed: str, target: str, phase: str) -> np.ndarray:
    import cv2
    img = np.ascontiguousarray(frame[:, :, ::-1])      # RGB -> BGR for cv2
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 64), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"typing: {typed}|", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    color = (0, 220, 0) if phase == "strike" else (0, 200, 255)
    cv2.putText(img, f"key '{target}'  [{phase}]", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--text", required=True, help="string to type")
    p.add_argument("--approach", default="models/approach_v16b_residual.pt",
                   help="residual-on-IK Approach checkpoint")
    p.add_argument("--tube", type=float, default=0.15, help="residual tube (match training)")
    p.add_argument("--keyboard-offset", default="-0.10,-0.10",
                   help="keyboard 'x,y' (match training; use the =form for the minus)")
    p.add_argument("--out", default="typing.mp4", help="output MP4 path")
    p.add_argument("--interactive", action="store_true", help="live viewer instead of MP4 (needs X11)")
    p.add_argument("--camera", default="agentview",
                   help="agentview (default; shows arm + keyboard) | frontview | "
                        "sideview | birdview")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-every", type=int, default=2, help="render every N sim steps")
    p.add_argument("--max-approach-steps", type=int, default=300)
    p.add_argument("--max-strike-steps", type=int, default=60)
    p.add_argument("--continuous", action="store_true",
                   help="flow key->key without resetting (smoother but less "
                        "reliable: the policy was trained from the home pose, so "
                        "starting each key from the previous key's pose is "
                        "out-of-distribution). Default resets to home per key.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from rl_autonomy.algos import RLPDSAC, RLPDConfig
    from rl_autonomy.envs import make_residual_env, KeyboardEnv, ResidualIKWrapper, AVAILABLE_KEYS
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.envs.action_adapter import ActionAdapter

    # resolve the character sequence up front (skip unmappable ones)
    seq: list[tuple[str, str]] = []
    for ch in args.text:
        key = char_to_key(ch)
        if key is None or key not in AVAILABLE_KEYS:
            print(f"[type] skipping {ch!r} (no key)")
            continue
        seq.append((ch, key))
    if not seq:
        print("[type] nothing typeable in --text"); return 1
    print(f"[type] typing {len(seq)} keys: {''.join(k for _, k in seq)}")

    kb_off = tuple(float(v) for v in args.keyboard_offset.split(","))
    render = not args.interactive
    # one long episode so the arm flows key->key without resetting
    horizon = (args.max_approach_steps + args.max_strike_steps + 20) * len(seq) + 100
    env = make_residual_env(tube=args.tube, reward_mode="xy_focus", keyboard_offset=kb_off,
                            random_key=False, frame_stack=3, seed=args.seed, horizon=horizon)
    kb = find_inner(env, KeyboardEnv)
    aa = find_inner(env, ActionAdapter)
    res = find_inner(env, ResidualIKWrapper)
    if args.interactive:
        kb.has_renderer = True   # best-effort; offscreen MP4 is the supported path

    agent = RLPDSAC(env=env, config=RLPDConfig(), device=args.device)
    agent.load(args.approach)

    frames: list[np.ndarray] = []
    typed = ""
    extend = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
    retract = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)

    def grab(target: str, phase: str):
        if not render:
            kb.render(); return
        f = kb.sim.render(width=args.width, height=args.height, camera_name=args.camera)[::-1]
        frames.append(_annotate(f, typed, target, phase))

    obs, _ = env.reset()
    n_pressed = 0
    for ci, (ch, key) in enumerate(seq):
        # ---- per-key reset to home (default) ----
        # The Approach policy was trained from the home init pose, so each key is
        # a fresh approach from home — this matches the 98.6% eval (TRACKER §39).
        # --continuous skips the reset (smoother, but OOD starts => less reliable).
        aa.set_mode("approach"); res.bypass = False; kb.mode = "approach"
        kb.set_target_key(key); kb.done = False
        if not (args.continuous and ci > 0):
            obs, _ = env.reset()
            kb.set_target_key(key); kb.done = False
        # ---- APPROACH ----
        reached = False
        for t in range(args.max_approach_steps):
            a, _ = agent.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            kb.done = False
            if t % args.frame_every == 0:
                grab(key, "approach")
            if info.get("is_success"):
                reached = True; break
        # ---- STRIKE (in-process, open-loop solenoid) ----
        aa.set_mode("strike"); res.bypass = True
        kb.mode = "strike"; kb._contact_steps = 0; kb.done = False
        pressed = False
        for t in range(args.max_strike_steps):
            obs, _, term, trunc, info = env.step(extend)
            kb.done = False
            if t % args.frame_every == 0:
                grab(key, "strike")
            if kb._check_success():
                pressed = True; break
        if pressed:
            typed += ch; n_pressed += 1
        # ---- RETRACT solenoid (continuous mode only; reset re-homes otherwise) ----
        if args.continuous:
            for t in range(10):
                obs, _, term, trunc, info = env.step(retract)
                kb.done = False
                if t % args.frame_every == 0:
                    grab(key, "lift")
        print(f"[type] {ci+1}/{len(seq)} key {key!r}: "
              f"{'reached' if reached else 'MISS'} / {'PRESSED' if pressed else 'no-press'}",
              flush=True)
    env.close()
    print(f"[type] typed {n_pressed}/{len(seq)} keys successfully")

    if args.interactive:
        return 0
    if not frames:
        print("[type] no frames rendered"); return 1
    import cv2
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             float(args.fps), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[type] wrote {len(frames)} frames -> {args.out}  "
          f"({len(frames)/args.fps:.1f}s @ {args.fps}fps)")
    print(f"[type] copy it out:  docker cp rover_gpu:{os.path.abspath(args.out)} .")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
