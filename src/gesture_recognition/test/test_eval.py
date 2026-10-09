"""Recorder -> .npz -> eval round trip on synthetic people."""
import numpy as np

from gesture_recognition.cli import Recorder
from gesture_recognition.eval import main as eval_main
from gesture_recognition.synthetic import as_persons, generate


def record(tmp_path, motion, label, seed):
    t, k, c, b = generate(right=motion, duration=6.0, fps=15, noise_px=1.0, seed=seed)
    rec = Recorder(tmp_path, label, f"synthetic_{seed}")
    for ti, p in zip(t, as_persons(k, c, b)):
        p.track_id = 0
        rec.add(float(ti), [p], (1280, 720))
    return rec.close()


def test_eval_round_trip(tmp_path, capsys):
    record(tmp_path, "beckon_up", "beckon", 0)
    record(tmp_path, "beckon_palmdown", "beckon", 1)
    record(tmp_path, "talk", "negative", 2)
    record(tmp_path, "wave", "wave", 3)
    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 4
    d = np.load(files[0])
    assert d["keypoints"].shape[1:] == (17, 2)

    eval_main([str(tmp_path)])
    import json
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["beckon_recall"] == 1.0
    assert summary["false_events_per_min"] == 0.0
