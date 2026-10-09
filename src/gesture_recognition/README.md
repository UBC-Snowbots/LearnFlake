# gesture_recognition

Detects a person **beckoning** ("come here") in the rover camera feed and publishes an event with
which way the person is. It also labels **waves**, so the two don't get confused. More gestures
come in Phase 3 (learned model).

Status: **Phase 1 baseline (rule-based)**. Tested on synthetic skeletons and on real stock
footage of people who are *not* beckoning (false-alarm check). It has **not** yet been tested on
real beckoning footage or on the rover camera; that is Phase 2.

## How it works

```
camera frame
  -> YOLO-pose on the full frame (17 body keypoints per person)
  -> "zoom" pass: far-away people are cropped from the full-res frame and run through pose again
  -> tracker (keeps each person's last ~3 s of keypoints)
  -> rule detector, 5x per second, on the last 2 s of each person   (detector.py)
  -> debounce: 3 of the last 5 decisions must agree, then a 4 s cooldown per person
  -> event: {"label": "beckon", "confidence", "track_id", "arm", "stamp", "bbox", "bearing_deg", ...}
```

**Why video and not single frames:** a beckon is a motion. In one still frame it looks the same
as a wave or a raised hand, so every decision looks at the last 2 s of keypoints.

**Why body pose and not hand landmarks:** published tests (Bamani et al. 2023,
arXiv 2311.15361) show hand-landmark tools such as MediaPipe fall apart past about 5-7 m. Body
keypoints (shoulder, elbow, wrist) survive much further, especially with the zoom pass, which is
a cheap version of their "enlarge the person first" trick.

**The rule detector** (`detector.py`) looks at the wrist relative to the shoulder, divided by
shoulder width (so walking, camera shake and distance cancel out), and splits the motion into:

| channel | what it measures | which gesture |
|---|---|---|
| vertical | wrist goes up/down | beckon (palm-up flex toward the face, palm-down scoop) |
| depth | apparent forearm length changes | beckon (forearm swinging toward the person, seen head-on) |
| lateral | wrist goes side to side | wave |

It calls **beckon** when vertical/depth motion beats lateral by 1.3x, and **wave** the other way
round. Both also need all of these:
- the hand above mid-torso
- at least 1.5 back-and-forth cycles at 0.6-4 Hz
- motion that fits a steady rhythm (sine-fit R^2 >= 0.6), so keypoint jitter on far, small
  people doesn't count

## Quick start (no ROS)

```bash
pip install -r requirements.txt            # ultralytics pulls torch; first run downloads yolo11n-pose.pt
cd src/gesture_recognition
python -m gesture_recognition.cli --source clip.mp4 --out annotated.mp4 --events events.jsonl
python -m gesture_recognition.cli --source 0 --show            # webcam, press q to quit
python -m pytest -q test                                       # 45 tests, no GPU or ROS needed
```

Useful flags: `--device 0` (GPU), `--hfov 70` (camera horizontal FOV for the bearing),
`--clips events/` (saves the ~3 s leading up to each event as an mp4), `--no-zoom`.

## Testing it on real people (do this first)

1. **Record clips** on a phone or the rover camera, 10-20 s each, landscape, whole body in frame.
   Name each one `<label>_<anything>.mp4`. Only `beckon` clips are expected to fire:

   | file name | what to do in it |
   |---|---|
   | `beckon_<who>_<dist>.mp4` | beckon "come here" with the whole forearm, 3+ times. Do palm-up and palm-down versions, each arm, at 2, 5 and 10 m |
   | `wave_<who>.mp4` | wave hello |
   | `goaway_<who>.mp4` | shooing / "go away" (expected to FAIL for now, see limitations) |
   | `none_<what>.mp4` | walking around, talking with hands, pointing, "stop" palm, phone, stretching |

2. **Copy them to the box** (from the laptop): `scp *.mp4 pranav@<box>:~/gesture_clips/`
3. **Run the batch test** (CPU, about real time at the default 15 fps cap):
   ```bash
   cd src/gesture_recognition
   python -m gesture_recognition.batch ~/gesture_clips
   ```
   It prints a PASS/FAIL table (also saved as `results/report.md`). For each clip it lists the
   most common reasons arms were rejected (e.g. `hand not raised`, `not repeated enough`,
   `motion direction ambiguous`), which tells you which threshold a miss ran into.
4. **Watch** `results/*.annotated.mp4`: every person shows their decision plus each arm's verdict
   or rejection reason and hand height `h` (0 = hip, 1 = shoulder; beckons need `h >= 0.5`).
5. **Retune without re-running video**: `python -m gesture_recognition.eval ~/gesture_clips/results/rec --min-raise 0.4`
   replays the saved keypoints with new thresholds in seconds.

For a live check on a laptop webcam instead: copy `src/gesture_recognition/` over,
`pip install ultralytics opencv-python`, then `python -m gesture_recognition.cli --source 0 --show`.

## ROS 2 (Humble)

```bash
pip install -r src/gesture_recognition/requirements.txt   # ultralytics is not in rosdep
colcon build --packages-select gesture_recognition && source install/setup.bash
ros2 launch gesture_recognition gesture.launch.py
ros2 topic echo /gesture/events
```

| | topic | type |
|---|---|---|
| in | `/camera/image_raw/compressed` (param `image_topic`) | `sensor_msgs/CompressedImage`, or `Image` if the name doesn't end in `/compressed` |
| out | `/gesture/events` | `std_msgs/String`, one JSON event per beckon |
| out | `/gesture/debug_image/compressed` | jpeg with skeletons and labels (only built when someone subscribes) |

`bearing_deg` follows ROS REP-103: positive = person is to the **left** of the camera axis.
Parameters are in `config/params.yaml`; **`hfov_deg` is a placeholder** until we know the camera.
The node keeps only the newest frame (sensor-data QoS), so a slow model drops frames instead of lagging.

**Not yet run under ROS**: the dev box has no ROS install. The node is a thin wrapper over the
tested pipeline, but it needs a first run on a Humble machine.

## Collecting data (Phase 2) and scoring

```bash
# record each clip with what the person is doing; one .npz per tracked person
python -m gesture_recognition.cli --source beckon_5m_alice.mp4 --record data/raw --label beckon
python -m gesture_recognition.cli --source walking_crowd.mp4  --record data/raw --label negative
# score the detector on everything recorded (fast: replays keypoints, no video/GPU)
python -m gesture_recognition.eval data/raw --min-raise 0.5
# -> {"beckon_recall": ..., "median_time_to_event_s": ..., "false_events_per_min": ...}
```

Every `DetectorConfig` field is a flag on `eval`, so thresholds can be swept on real data.
What to record: 10+ people, 2-15 m, outdoors, different lighting. Include palm-up **and**
palm-down beckons, with either arm. Record **lots** of negatives: waving, pointing, "stop",
"go away", walking, talking with hands, phone use.

## Known limitations

- **Untested on real beckons.** Thresholds (`min_raise`, `min_amplitude`, `dominance`,
  `min_periodicity`) were set on synthetic skeletons and are guesses until Phase 2 data.
- **"Go away" (shooing) moves on the same axis as a beckon** and will likely fire. Separating
  them needs stroke direction or palm orientation: Phase 3.
- **Finger-only or wrist-only beckons** (forearm still) are invisible to body keypoints. They
  need hand landmarks at close range.
- **Low beckons** (hand below mid-torso) are ignored on purpose, to suppress talking-with-hands.
- If a person's track is lost and re-acquired, the cooldown resets and the event can fire again.
- A side-on person shows little lateral motion, so a wave seen from the side can look like a
  beckon.

## Roadmap

1. ~~Rule baseline + ROS node~~ (this)
2. Record real data with `--record`; score with `eval`; retune thresholds
3. Small temporal model (1D CNN/GRU) on keypoint sequences: classes beckon / wave / stop / point / go-away / none
4. Evaluate on held-out people: recall per distance, false events per 10 min, latency
5. Deploy on the rover computer (profile; TensorRT export via `yolo export format=engine` if needed)

## Files

| file | what |
|---|---|
| `pose.py` | YOLO-pose + zoom pass |
| `tracker.py` | IoU tracker + per-person keypoint history |
| `detector.py` | rule detector (beckon / wave / none) |
| `events.py` | debounce, cooldown, event record, bearing |
| `pipeline.py` | ties it together (no ROS) |
| `cli.py` | video/webcam runner, event clips, `--record` |
| `eval.py` | scores recorded data |
| `batch.py` | pass/fail test on a folder of labelled clips |
| `synthetic.py` | stick-figure motions for tests |
| `ros_node.py`, `launch/`, `config/` | ROS 2 wrapper |
