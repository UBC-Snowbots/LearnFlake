"""Run the detector on a video file, webcam, or stream URL (no ROS needed).

  gesture-detect --source clip.mp4 --out annotated.mp4 --events events.jsonl
  gesture-detect --source 0 --show                              # webcam
  gesture-detect --source clip.mp4 --record data/raw --label beckon   # save keypoints for training
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig
from .events import DebounceConfig
from .pipeline import GesturePipeline
from .pose import YoloPoseEstimator
from .viz import draw


def build_pipeline(args) -> GesturePipeline:
    est = YoloPoseEstimator(model=args.model, device=args.device, imgsz=args.imgsz, zoom=not args.no_zoom,
                            threads=args.threads)
    return GesturePipeline(est, DetectorConfig(window_s=args.window), DebounceConfig(cooldown_s=args.cooldown),
                           hfov_deg=args.hfov, eval_hz=args.eval_hz)


class Recorder:
    """Keeps every tracked person's full keypoint sequence; saves one .npz per person on close."""

    def __init__(self, out_dir: Path, label: str, source: str, min_s: float = 1.0):
        self.out_dir, self.label, self.source, self.min_s = out_dir, label, source, min_s
        self.seqs: dict[int, dict[str, list]] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, t: float, persons, image_size) -> None:
        for p in persons:
            s = self.seqs.setdefault(p.track_id, {"t": [], "kpts": [], "conf": [], "bbox": []})
            s["t"].append(t)
            s["kpts"].append(p.keypoints)
            s["conf"].append(p.confidence)
            s["bbox"].append(p.bbox)
        self.image_size = image_size

    def close(self) -> list[Path]:
        saved = []
        stem = Path(self.source).stem if not self.source.isdigit() else f"cam{self.source}"
        for tid, s in self.seqs.items():
            if len(s["t"]) < 2 or s["t"][-1] - s["t"][0] < self.min_s:
                continue
            path = self.out_dir / f"{stem}_{int(time.time())}_track{tid}_{self.label}.npz"
            np.savez_compressed(path, t=np.array(s["t"]), keypoints=np.stack(s["kpts"]),
                                confidence=np.stack(s["conf"]), bbox=np.stack(s["bbox"]), label=self.label,
                                source=self.source, image_size=np.array(self.image_size))
            saved.append(path)
        return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, camera index, or stream URL")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="ultralytics pose weights (auto-downloads)")
    ap.add_argument("--device", default=None, help="cpu, 0 (GPU), ... default: auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-zoom", action="store_true", help="disable the far-person crop pass")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (keep <= 8 on the shared box)")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view, degrees")
    ap.add_argument("--window", type=float, default=2.0, help="seconds of motion per decision")
    ap.add_argument("--eval-hz", type=float, default=5.0)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--out", help="write annotated video here")
    ap.add_argument("--events", help="append events as JSON lines here")
    ap.add_argument("--clips", help="save a short annotated clip of each event into this folder")
    ap.add_argument("--record", help="save per-person keypoint sequences (.npz) into this folder")
    ap.add_argument("--label", default="unlabeled", help="label stored with --record")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"cannot open {args.source}", file=sys.stderr)
        return 2
    is_file = isinstance(src, str) and Path(src).exists()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe = build_pipeline(args)
    writer = None
    events_f = open(args.events, "a") if args.events else None
    recorder = Recorder(Path(args.record), args.label, args.source) if args.record else None
    clip_buf: deque = deque(maxlen=int(fps * (args.window + 1.0)))
    n, n_events, t_start, infer_s = 0, 0, time.monotonic(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            t = n / fps if is_file else time.monotonic() - t_start  # video time for files
            t0 = time.perf_counter()
            out = pipe.process(frame, t)
            infer_s += time.perf_counter() - t0
            n += 1
            if recorder:
                recorder.add(t, out.persons, (frame.shape[1], frame.shape[0]))
            need_vis = args.out or args.show or args.clips
            vis = draw(frame, out) if need_vis else None
            if args.clips:
                clip_buf.append(vis)
            for ev in out.events:
                n_events += 1
                line = ev.to_json()
                print(line, flush=True)
                if events_f:
                    events_f.write(line + "\n")
                    events_f.flush()
                if args.clips:
                    Path(args.clips).mkdir(parents=True, exist_ok=True)
                    path = Path(args.clips) / f"event_{n_events:03d}_{ev.label}_t{ev.stamp:.1f}.mp4"
                    h, w = vis.shape[:2]
                    cw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    for f in clip_buf:
                        cw.write(f)
                    cw.release()
            if args.out:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)
            if args.show:
                cv2.imshow("gesture", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if events_f:
            events_f.close()
        if args.show:
            cv2.destroyAllWindows()
    summary = {"frames": n, "events": n_events, "ms_per_frame": round(1000 * infer_s / max(n, 1), 1)}
    if recorder:
        summary["recorded"] = [str(p) for p in recorder.close()]
    print(json.dumps({"summary": summary}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
