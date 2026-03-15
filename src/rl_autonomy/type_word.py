#!/usr/bin/env python3
"""
Keyboard Typing Orchestrator — strings CoarseReach + FineAlign + solenoid
press together to type a word in simulation.

Usage:
    python type_word.py --word hello
    python type_word.py --word hello --no-render
    python type_word.py --word "the quick brown fox" --coarse checkpoints/coarse_reach/best_model.zip --fine checkpoints/fine_align/best_model.zip
"""

import os
import sys
import argparse
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

from stable_baselines3 import SAC
from keyboard_env import KeyboardEnv, CoarseReachEnv, FineAlignEnv


# ============================================================================
# Character → key name mapping
# ============================================================================

CHAR_TO_KEY = {
    # Letters
    **{c: c for c in "abcdefghijklmnopqrstuvwxyz"},
    # Numbers
    **{c: c for c in "1234567890"},
    # Punctuation
    "`": "grave", "~": "grave",
    "-": "minus", "_": "minus",
    "=": "equal", "+": "equal",
    "[": "lbracket", "{": "lbracket",
    "]": "rbracket", "}": "rbracket",
    "\\": "backslash", "|": "backslash",
    ";": "semicolon", ":": "semicolon",
    "'": "quote", '"': "quote",
    ",": "comma", "<": "comma",
    ".": "period", ">": "period",
    "/": "slash", "?": "slash",
    " ": "space",
}


# ============================================================================
# Orchestrator
# ============================================================================

class KeyboardOrchestrator:
    """Rule-based state machine that chains skills to type a word."""

    # Success thresholds (match training envs)
    COARSE_XY   = CoarseReachEnv.SUCCESS_XY      # 3 cm
    COARSE_Z    = CoarseReachEnv.SUCCESS_Z
    COARSE_TILT = CoarseReachEnv.SUCCESS_TILT
    FINE_XY     = FineAlignEnv.SUCCESS_XY         # 5 mm
    FINE_Z      = FineAlignEnv.SUCCESS_Z
    FINE_TILT   = FineAlignEnv.SUCCESS_TILT
    HOVER_HEIGHT = CoarseReachEnv.HOVER_HEIGHT

    # Step limits per phase
    COARSE_MAX_STEPS = 300
    FINE_MAX_STEPS   = 200
    PRESS_STEPS      = 15    # solenoid extend duration
    RETRACT_STEPS    = 10    # solenoid retract duration

    def __init__(self, coarse_model_path: str, fine_model_path: str,
                 render: bool = True):
        self.env = KeyboardEnv(render=render, horizon=100_000)
        self.env.reset()
        if render:
            self.env.render()

        self.coarse_policy = SAC.load(coarse_model_path)
        self.fine_policy   = SAC.load(fine_model_path)
        self.render_enabled = render

    def _get_obs(self):
        return self.env._flat_obs().astype(np.float32)

    def _get_metrics(self):
        """Return (xy_dist, z_error, tilt) for current EEF vs target key."""
        sim = self.env.sim
        eef_pos = sim.data.site_xpos[self.env._eef_site_id]
        key_pos = sim.data.body_xpos[self.env._key_body_ids[self.env._target_key]]

        pf = self.env.robots[0].robot_model.naming_prefix
        eef_quat = self.env._get_observations(force_update=False).get(
            f"{pf}eef_quat", np.array([1, 0, 0, 0])
        )

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(KeyboardEnv._eef_tilt_from_vertical(eef_quat))
        return xy_dist, z_error, tilt

    def _step(self, action_6d, solenoid=0.0):
        """Step the env with 6D arm action + solenoid command."""
        full_action = np.append(action_6d, solenoid)
        _, reward, done, info = self.env.step(full_action)
        if self.render_enabled:
            self.env.render()
        return reward, done

    def _find_closest_key(self):
        """Return the name of the key closest to the solenoid tip in XY."""
        sim = self.env.sim
        tip_id = sim.model.site_name2id("gripper0_right_actuator_tip_site")
        tip_pos = sim.data.site_xpos[tip_id]
        best_key = ""
        best_dist = float("inf")
        for name in self.env.AVAILABLE_KEYS:
            kpos = sim.data.body_xpos[self.env._key_body_ids[name]]
            d = float(np.linalg.norm(tip_pos[:2] - kpos[:2]))
            if d < best_dist:
                best_dist = d
                best_key = name
        return best_key, best_dist

    # ------------------------------------------------------------------
    # Skill phases
    # ------------------------------------------------------------------

    def run_coarse_reach(self, key_name: str):
        """Run CoarseReach policy until success or timeout."""
        self.env.set_target_key(key_name)
        obs = self._get_obs()

        for step in range(self.COARSE_MAX_STEPS):
            action, _ = self.coarse_policy.predict(obs, deterministic=True)
            self._step(action)
            obs = self._get_obs()

            xy, ze, tilt = self._get_metrics()
            if xy < self.COARSE_XY and ze < self.COARSE_Z and tilt < self.COARSE_TILT:
                return True, step + 1, xy, ze, tilt

        xy, ze, tilt = self._get_metrics()
        return False, self.COARSE_MAX_STEPS, xy, ze, tilt

    def run_fine_align(self, key_name: str):
        """Run FineAlign policy until success or timeout."""
        self.env.set_target_key(key_name)
        obs = self._get_obs()

        for step in range(self.FINE_MAX_STEPS):
            action, _ = self.fine_policy.predict(obs, deterministic=True)
            # FineAlign uses 0.3x action scaling
            self._step(action * 0.3)
            obs = self._get_obs()

            xy, ze, tilt = self._get_metrics()
            if xy < self.FINE_XY and ze < self.FINE_Z and tilt < self.FINE_TILT:
                return True, step + 1, xy, ze, tilt

        xy, ze, tilt = self._get_metrics()
        return False, self.FINE_MAX_STEPS, xy, ze, tilt

    def run_press(self):
        """Fire solenoid for PRESS_STEPS, holding arm still."""
        for _ in range(self.PRESS_STEPS):
            self._step(np.zeros(6), solenoid=1.0)
        pressed_key, press_dist = self._find_closest_key()
        return pressed_key, press_dist

    def run_retract(self):
        """Retract solenoid for RETRACT_STEPS."""
        for _ in range(self.RETRACT_STEPS):
            self._step(np.zeros(6), solenoid=0.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def type_word(self, word: str):
        """Type a word by chaining skills for each character."""
        # Convert characters to key names
        keys = []
        for ch in word.lower():
            key_name = CHAR_TO_KEY.get(ch)
            if key_name is None:
                print(f"  WARNING: no key mapping for '{ch}', skipping")
                continue
            if key_name not in self.env.AVAILABLE_KEYS:
                print(f"  WARNING: key '{key_name}' not in keyboard layout, skipping")
                continue
            keys.append((ch, key_name))

        print(f"\n{'='*60}")
        print(f"  TYPING: \"{word}\"")
        print(f"  Keys:   {[k for _, k in keys]}")
        print(f"{'='*60}\n")

        results = []
        total_start = time.time()

        for i, (char, key_name) in enumerate(keys):
            print(f"--- Letter {i+1}/{len(keys)}: '{char}' (key={key_name}) ---")

            # Phase 1: CoarseReach
            ok, steps, xy, ze, tilt = self.run_coarse_reach(key_name)
            status = "OK" if ok else "TIMEOUT"
            print(f"  CoarseReach: {status} in {steps} steps  "
                  f"xy={xy*100:.2f}cm  z_err={ze*100:.2f}cm  tilt={np.degrees(tilt):.1f}°")

            # Phase 2: FineAlign
            ok, steps, xy, ze, tilt = self.run_fine_align(key_name)
            status = "OK" if ok else "TIMEOUT"
            print(f"  FineAlign:   {status} in {steps} steps  "
                  f"xy={xy*100:.2f}cm  z_err={ze*100:.2f}cm  tilt={np.degrees(tilt):.1f}°")

            # Phase 3: Press
            pressed, dist = self.run_press()
            hit = pressed == key_name
            print(f"  Press:       solenoid fired → closest key='{pressed}' "
                  f"(dist={dist*100:.2f}cm)  {'HIT' if hit else 'MISS (wanted: ' + key_name + ')'}")

            # Phase 4: Retract
            self.run_retract()
            print(f"  Retract:     done\n")

            results.append({
                "char": char,
                "target": key_name,
                "pressed": pressed,
                "hit": hit,
                "final_xy": xy,
                "final_z": ze,
                "press_dist": dist,
            })

        # Summary
        elapsed = time.time() - total_start
        hits = sum(1 for r in results if r["hit"])
        typed = "".join(r["char"] if r["hit"] else "_" for r in results)

        print(f"{'='*60}")
        print(f"  RESULT: \"{typed}\"  ({hits}/{len(results)} correct)")
        print(f"  Time:   {elapsed:.1f}s")
        print(f"{'='*60}")

        for r in results:
            mark = "OK" if r["hit"] else "MISS"
            print(f"    '{r['char']}' → target={r['target']:10s}  "
                  f"pressed={r['pressed']:10s}  "
                  f"align_xy={r['final_xy']*100:.2f}cm  {mark}")

        return results

    def close(self):
        self.env.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Type a word in MuJoCo simulation")
    parser.add_argument("--word", type=str, default="hi",
                        help="Word to type (default: 'hi')")
    parser.add_argument("--coarse", type=str,
                        default=os.path.join(ROOT, "checkpoints", "coarse_reach", "best_model.zip"),
                        help="CoarseReach checkpoint path")
    parser.add_argument("--fine", type=str,
                        default=os.path.join(ROOT, "checkpoints", "fine_align", "best_model.zip"),
                        help="FineAlign checkpoint path")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    orch = KeyboardOrchestrator(
        coarse_model_path=args.coarse,
        fine_model_path=args.fine,
        render=not args.no_render,
    )

    try:
        orch.type_word(args.word)
    finally:
        orch.close()


if __name__ == "__main__":
    main()
