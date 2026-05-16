#!/bin/bash
# Quick status snapshot for any v1 training run.
#
# Usage:
#   ./scripts/training_status.sh              # default: approach_v1
#   ./scripts/training_status.sh strike_v1    # check Strike
#   ./scripts/training_status.sh approach_v1_attempt2   # any archived run
#
# Works from host or from inside the rover_gpu container.
set -u

RUN="${1:-approach_v1}"

# Match the python process name based on the run prefix
case "$RUN" in
    approach*) PROC_PAT="train_approach" ;;
    strike*)   PROC_PAT="train_strike"   ;;
    *)         PROC_PAT="train_$(echo "$RUN" | cut -d_ -f1)" ;;
esac

# Path resolution: host vs container
if [[ -f /.dockerenv ]]; then
    REPO="/LearnFlake"
else
    REPO="/home/arhim/Documents/rover/LearnFlake"
fi
LOG="$REPO/logs/$RUN/train.log"
CKPT_DIR="$REPO/checkpoints/$RUN"

echo "=== ${RUN} training status ==="
echo "log:  $LOG"
echo "ckpt: $CKPT_DIR"
echo

# 1. Is the python process alive?
if pgrep -f "$PROC_PAT" > /dev/null 2>&1; then
    echo "[proc] training process ($PROC_PAT): ALIVE"
    ps -ef | grep "$PROC_PAT" | grep -v grep | awk '{print "  pid="$2"  cpu="$3"%  mem="$4"%  elapsed="$NF}'
else
    echo "[proc] training process ($PROC_PAT): NOT FOUND"
    echo "       (run might be finished, or you might need to check inside rover_gpu)"
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
    echo "[log] eval trajectory (all):"
    grep 'eval_return=' "$LOG" | sed 's/^/  /' || echo "  (none yet)"
    echo
    cur_evt=$(grep -E 'curriculum advanced|curriculum reached' "$LOG" | tail -3)
    if [[ -n "$cur_evt" ]]; then
        echo "[log] curriculum:"
        echo "$cur_evt" | sed 's/^/  /'
        echo
    fi
    # Real post-warmup nan only (not the early warmstart formatting)
    nan_lines=$(grep -E 'Traceback|RuntimeError|Killed' "$LOG" || true)
    if [[ -n "$nan_lines" ]]; then
        echo "[log] CRASHES:"
        echo "$nan_lines" | tail -3 | sed 's/^/  /'
    fi
else
    echo "[log] LOG NOT FOUND at $LOG"
fi

# 4. Checkpoint count
if [[ -d "$CKPT_DIR" ]]; then
    n=$(ls -1 "$CKPT_DIR"/*.pt 2>/dev/null | wc -l)
    echo
    echo "[ckpt] $n checkpoint(s):"
    ls -lh "$CKPT_DIR"/*.pt 2>/dev/null | awk '{print "  "$NF, "("$5")"}' | tail -6 || echo "  (no checkpoints yet)"
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
