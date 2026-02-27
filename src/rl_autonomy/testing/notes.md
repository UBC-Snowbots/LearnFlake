## Step 1 — Record initial demos

You need **two terminals inside Docker**:

**Terminal A** — MuJoCo bridge (receives joystick twist from RoverFlake2):
```bash
cd /LearnFlake/src/rl_autonomy/testing
python3 cartesian_control_ros.py
```

**Terminal B** — Demo recorder (records what the bridge does):
```bash
cd /LearnFlake/src/rl_autonomy/testing
python3 demo_recorder.py
```

Then drive the arm with the joystick. Press **Enter** after each successful episode (e.g. reach→grasp→lift). Press **d** to discard a bad one. **Ctrl+C** when done — it writes `demos/demos_TIMESTAMP.hdf5`.

Aim for ~5-10 demos to start.

## Step 2 — Train BC

```bash
cd /LearnFlake/src/rl_autonomy/testing
python3 bc_train.py demos/*.hdf5 --epochs 100 --device cuda
```

This trains a small MLP (256,256) and saves:
- `models/bc_best.pt` (lowest loss)
- `models/bc_latest.pt` (final epoch)

## Step 3 — Evaluate

```bash
python3 bc_train.py --eval models/bc_best.pt --eval-episodes 5 --device cuda
```

This opens MuJoCo and runs the policy for 5 episodes. You'll see success/fail + reward for each.

## Step 4 — DAgger iteration (optional but recommended)

If the policy is shaky, collect corrections:

```bash
python3 dagger_collect.py --policy models/bc_best.pt
```

This runs the policy autonomously. **Touch the joystick to override** whenever it does something wrong — your corrections get recorded. Press Enter between episodes, Ctrl+C to save (`demos/dagger_TIMESTAMP.hdf5`).

Then **retrain on everything** (initial + DAgger data aggregated):

```bash
python3 bc_train.py demos/*.hdf5 --epochs 200 --device cuda
```

Repeat steps 3-4 until it looks good.

---

**Note:** Step 1 requires the RoverFlake2 arm launch running (for the twist topic). Steps 2-4 only need the LearnFlake Docker container. For eval/DAgger you need `DISPLAY=:1` and `xhost +local:docker` on the host for the MuJoCo window.

Made changes.



cd /LearnFlake/src/rl_autonomy/testing

# Convert BC to HRL-SAC warm start
python3 bc_to_rl.py --bc-checkpoint models/bc_best.pt --output models/bc_warm_start_v2.pt

# Fine-tune with HRL-SAC from that warm start
python3 train_lift_v2.py --train --cuda --resume models/bc_warm_start_v2.pt --hidden 512 512 256


# maximum transfer
python3 bc_to_rl.py --bc-checkpoint models/bc_best.pt --output models/bc_warm_start_match.pt --match-bc-hidden
python3 train_lift_v2.py --train --cuda --resume models/bc_warm_start_match.pt --hidden 256 256
