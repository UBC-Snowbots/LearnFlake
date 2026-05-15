#!/bin/bash
# Quick status snapshot for the M3 Approach training run.
# Run from host: ./scripts/training_status.sh
# Run inside container: bash /LearnFlake/scripts/training_status.sh

set -u
LOG="/home/arhim/Documents/rover/LearnFlake/logs/approach_v1/train.log"
CKPT_DIR="/home/arhim/Documents/rover/LearnFlake/checkpoints/approach_v1"

if [[ -f /.dockerenv ]]; then
    LOG="/LearnFlake/logs/approach_v1/train.log"
    CKPT_DIR="/LearnFlake/checkpoints/approach_v1"
fi

echo "=== M3 Approach training status ==="
echo "log:  $LOG"
echo "ckpt: $CKPT_DIR"
echo

# 1. Is the python process alive?
if pgrep -f 'train_approach' > /dev/null 2>&1; then
    echo "[proc] training process: ALIVE"
    echo "  $(ps -ef | grep train_approach | grep -v grep | awk '{print "  pid="$2"  cpu="$3"%  mem="$4"%  elapsed="$NF}')"
else
    echo "[proc] training process: NOT FOUND (might be on host shell — check rover_gpu container)"
fi

# 2. nvidia-smi quick
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'
fi

# 3. Last few significant log lines
if [[ -f "$LOG" ]]; then
    echo
    echo "[log] last step line:"
    grep '\[step' "$LOG" | tail -1 | sed 's/^/  /' || echo "  (no step lines yet)"
    echo
    echo "[log] last eval:"
    grep 'eval_return=' "$LOG" | tail -1 | sed 's/^/  /' || echo "  (no eval lines yet)"
    echo
    echo "[log] curriculum:"
    grep -E 'curriculum advanced|curriculum reached' "$LOG" | tail -3 | sed 's/^/  /' || echo "  (no curriculum events yet)"
    echo
    if grep -qE 'Traceback|RuntimeError|Killed|nan' "$LOG"; then
        echo "[log] WARNINGS — last 3:"
        grep -E 'Traceback|RuntimeError|Killed|nan' "$LOG" | tail -3 | sed 's/^/  /'
    fi
fi

# 4. Checkpoint count
if [[ -d "$CKPT_DIR" ]]; then
    n=$(ls -1 "$CKPT_DIR"/*.pt 2>/dev/null | wc -l)
    echo
    echo "[ckpt] $n checkpoint(s):"
    ls -lh "$CKPT_DIR"/*.pt 2>/dev/null | awk '{print "  "$NF, "("$5")"}' | tail -5 || echo "  (no checkpoints yet)"
fi

# 5. Wallclock summary
if [[ -f "$LOG" ]]; then
    start=$(head -1 "$LOG" | grep -oE '20[0-9-]+T[0-9:]+Z' || echo "")
    echo
    if [[ -n "$start" ]]; then
        echo "[time] started: $start"
        now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        start_s=$(date -d "$start" +%s 2>/dev/null || echo 0)
        now_s=$(date -d "$now" +%s 2>/dev/null || echo 0)
        if [[ "$start_s" -gt 0 && "$now_s" -gt 0 ]]; then
            elapsed=$(( now_s - start_s ))
            h=$(( elapsed / 3600 ))
            m=$(( (elapsed % 3600) / 60 ))
            echo "[time] elapsed: ${h}h ${m}m"
        fi
    fi
fi
