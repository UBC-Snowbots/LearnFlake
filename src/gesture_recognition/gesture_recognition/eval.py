"""Score the detector on recorded keypoint sequences (.npz from `gesture-detect --record`).

  python -m gesture_recognition.eval data/raw [--min-raise 0.5 ...]

Each file is one person; its `label` says what they were doing ("beckon" or anything else).
Reports: beckon recall (files where at least one beckon event fired), time to first event, and
false beckon events per minute on everything that is not a beckon. No video or GPU needed, so
threshold sweeps are cheap.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from .detector import DetectorConfig
from .events import DebounceConfig
from .keypoints import PersonPose
from .pipeline import GesturePipeline


def replay(path: Path, det: DetectorConfig, deb: DebounceConfig) -> dict:
    d = np.load(path, allow_pickle=False)
    pipe = GesturePipeline(estimator=None, detector=det, debounce=deb)
    size = tuple(int(v) for v in d["image_size"]) if "image_size" in d else (1920, 1080)
    events = []
    for t, k, c, b in zip(d["t"], d["keypoints"], d["confidence"], d["bbox"]):
        out = pipe.process_poses([PersonPose(bbox=b, keypoints=k, confidence=c)], float(t), size)
        events += [e for e in out.events if e.label == "beckon"]
    dur = float(d["t"][-1] - d["t"][0])
    return {"file": path.name, "label": str(d["label"]), "duration_s": round(dur, 2), "events": len(events),
            "first_event_s": round(events[0].stamp - float(d["t"][0]), 2) if events else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    for f in fields(DetectorConfig):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--cooldown", type=float, default=4.0)
    ap.add_argument("--json", action="store_true", help="print per-file rows too")
    args = ap.parse_args(argv)
    det = DetectorConfig(**{f.name: getattr(args, f.name) for f in fields(DetectorConfig)})
    deb = DebounceConfig(cooldown_s=args.cooldown)

    rows = [replay(p, det, deb) for p in sorted(Path(args.folder).glob("*.npz"))]
    pos = [r for r in rows if r["label"] == "beckon"]
    neg = [r for r in rows if r["label"] != "beckon"]
    neg_min = sum(r["duration_s"] for r in neg) / 60.0
    summary = {
        "files": len(rows),
        "beckon_recall": round(sum(r["events"] > 0 for r in pos) / len(pos), 3) if pos else None,
        "median_time_to_event_s": (float(np.median([r["first_event_s"] for r in pos if r["events"]]))
                                   if any(r["events"] for r in pos) else None),
        "false_events_per_min": round(sum(r["events"] for r in neg) / neg_min, 3) if neg_min else None,
        "negative_minutes": round(neg_min, 2),
    }
    if args.json:
        for r in rows:
            print(json.dumps(r))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
