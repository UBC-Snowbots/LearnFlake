"""Test the detector on a folder of labelled clips and print a pass/fail table.

  python -m gesture_recognition.batch clips/            # results go to clips/results/

Name each clip <label>_<anything>.mp4, e.g. beckon_alice_5m.mp4, wave_bob.mp4, none_walking.mp4,
goaway_carol.mp4. A "beckon" clip passes if at least one beckon event fires; every other clip
passes if none fires. For each clip you also get an annotated video (skeletons, the decision and
the reason for each arm), the events as JSON lines, and the keypoints as .npz for `eval` sweeps.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from .cli import Recorder
from .detector import NONE, DetectorConfig
from .events import DebounceConfig
from .pipeline import GesturePipeline
from .pose import YoloPoseEstimator
from .viz import Overlay

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def clip_label(path: Path) -> str:
    return path.stem.replace("-", "_").split("_")[0].lower()


def run_clip(path: Path, est, args, out_dir: Path) -> dict:
    label = clip_label(path)
    pipe = GesturePipeline(est, DetectorConfig(window_s=args.window, min_raise=args.min_raise),
                           DebounceConfig(cooldown_s=args.cooldown), hfov_deg=args.hfov)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / args.fps_cap)) if args.fps_cap else 1  # phone clips are 30-60 fps
    overlay, writer, rec = Overlay(), None, Recorder(out_dir / "rec", label, str(path))
    events, reasons, heights = [], Counter(), []
    n_read, n_proc, infer_s = 0, 0, 0.0
    with open(out_dir / f"{path.stem}.events.jsonl", "w") as ev_f:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_seconds and n_read / fps >= args.max_seconds):
                break
            n_read += 1
            if (n_read - 1) % step:
                continue
            t = (n_read - 1) / fps
            t0 = time.perf_counter()
            out = pipe.process(frame, t)
            infer_s += time.perf_counter() - t0
            n_proc += 1
            rec.add(t, out.persons, (frame.shape[1], frame.shape[0]))
            heights += [p.height_px for p in out.persons]
            for dec in out.decisions.values():
                for r in dec.arms.values():
                    if r.label == NONE and r.details.get("raise", 1.0) >= 0.0:  # skip arms hanging at rest
                        reasons[r.reason] += 1
            for ev in out.events:
                events.append(ev)
                ev_f.write(ev.to_json() + "\n")
            vis = overlay(frame, out, t)
            scale = min(1.0, args.video_width / vis.shape[1])
            if scale < 1.0:
                vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{path.stem}.annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
    cap.release()
    if writer:
        writer.release()
    rec.close()
    beckons = [e for e in events if e.label == "beckon"]
    expect_beckon = label == "beckon"
    passed = bool(beckons) if expect_beckon else not beckons
    return {
        "clip": path.name, "label": label, "pass": passed, "beckon_events": len(beckons),
        "first_event_s": round(beckons[0].stamp, 2) if beckons else None,
        "seconds": round(n_read / fps, 1), "person_px": round(max(heights)) if heights else 0,
        "ms_per_frame": round(1000 * infer_s / max(n_proc, 1), 1),
        "top_reasons": [f"{k} x{v}" for k, v in reasons.most_common(2)],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", help="results folder (default: <folder>/results)")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default="cpu", help="cpu (default) or 0 for GPU")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (keep <= 8 on the shared box)")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--fps-cap", type=float, default=15.0, help="process at most this many frames/s (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=0, help="only the first N seconds of each clip")
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--min-raise", type=float, default=0.5)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--video-width", type=int, default=1280, help="annotated video width")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    out_dir = Path(args.out) if args.out else folder / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        print(f"no videos in {folder}")
        return 2
    est = YoloPoseEstimator(model=args.model, device=args.device, zoom=not args.no_zoom, threads=args.threads)

    rows = []
    for p in clips:
        print(f"... {p.name}", flush=True)
        rows.append(run_clip(p, est, args, out_dir))

    pos = [r for r in rows if r["label"] == "beckon"]
    neg = [r for r in rows if r["label"] != "beckon"]
    neg_min = sum(r["seconds"] for r in neg) / 60
    lines = ["| clip | label | result | beckon events | first at (s) | length (s) | person px | ms/frame | top reasons (no-gesture arms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['clip']} | {r['label']} | {'PASS' if r['pass'] else 'FAIL'} | {r['beckon_events']} | "
                     f"{r['first_event_s'] if r['first_event_s'] is not None else '-'} | {r['seconds']} | "
                     f"{r['person_px']} | {r['ms_per_frame']} | {'; '.join(r['top_reasons'])} |")
    summary = {
        "clips": len(rows), "passed": sum(r["pass"] for r in rows),
        "beckon_clips_caught": f"{sum(r['pass'] for r in pos)}/{len(pos)}",
        "false_beckon_events": sum(r["beckon_events"] for r in neg),
        "false_events_per_min": round(sum(r["beckon_events"] for r in neg) / neg_min, 2) if neg_min else None,
    }
    report = "\n".join(lines) + "\n\n" + json.dumps(summary) + "\n"
    (out_dir / "report.md").write_text(report)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
    print(report)
    print(f"annotated videos, events and keypoints in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
