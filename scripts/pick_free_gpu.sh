#!/bin/bash
# Pin the current process to a genuinely-idle GPU, then exec the given command.
#
# Why this exists: the cluster runs with ConstrainDevices=no, so Slurm's GPU
# assignment is advisory -- it does not fence a job to its card. On a busy node,
# another user's process can already occupy the GPU Slurm handed us, and vLLM (which
# reserves gpu_memory_utilization, 0.90, at startup) then aborts with
# "Free memory ... less than desired GPU memory utilization". That is what failed 12
# of the Latxa ablation tasks.
#
# This picks a card with enough free memory *at launch* and pins to it via
# CUDA_VISIBLE_DEVICES, keeping utilization at 0.90 (so the KV cache -- and therefore
# the measured latency -- is unchanged). If no card is free yet, it waits: another
# user's job will finish, rather than us failing outright.
#
# Usage:  scripts/pick_free_gpu.sh <min_free_MiB> <command...>

set -euo pipefail

MIN_FREE_MIB="${1:-40000}"   # need ~40 GiB for a 0.90 reservation on a 46 GiB card
shift

MAX_WAIT_S=3600              # wait up to 1h for a card to free up
POLL_S=30
waited=0

# Two of our tasks run concurrently (%2). If both query nvidia-smi at the same
# instant they can pick the SAME free card and then both try to reserve 90% of it,
# reproducing the failure. A short-lived per-GPU lock makes each concurrent task
# claim a different card: a task holds the lock only until vLLM has actually grabbed
# the memory (a marker file it touches), after which the lock is released for reuse.
LOCK_DIR="${GPU_LOCK_DIR:-/tmp/gpu_locks_$USER}"
mkdir -p "$LOCK_DIR"

claim_gpu () {
  local idx="$1"
  # Atomic claim: mkdir succeeds for exactly one racer. Stale locks (>10 min, e.g.
  # a crashed task) are reclaimed so a dead lock can't wedge the queue forever.
  local lock="$LOCK_DIR/gpu${idx}.lock"
  if [[ -d "$lock" ]]; then
    local age=$(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || echo 0) ))
    (( age > 600 )) && rmdir "$lock" 2>/dev/null || true
  fi
  mkdir "$lock" 2>/dev/null
}

while true; do
  # Consider cards ordered by free memory, and claim the first free-enough one whose
  # lock we can take.
  while IFS=',' read -r idx free; do
    idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
    [[ "$free" -ge "$MIN_FREE_MIB" ]] || continue
    if claim_gpu "$idx"; then
      export CUDA_VISIBLE_DEVICES="$idx"
      export GPU_CLAIM_LOCK="$LOCK_DIR/gpu${idx}.lock"
      echo "[pick_free_gpu] claimed GPU ${idx} (${free} MiB free >= ${MIN_FREE_MIB})"
      # Release the lock ~90s later: enough for vLLM to reserve its memory, after
      # which this card's occupancy is real and the next task will see it as busy.
      ( sleep 90; rmdir "$GPU_CLAIM_LOCK" 2>/dev/null || true ) &
      exec "$@"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t',' -k2 -n -r)

  if [[ "$waited" -ge "$MAX_WAIT_S" ]]; then
    echo "[pick_free_gpu] no free+unlocked GPU with ${MIN_FREE_MIB} MiB after ${MAX_WAIT_S}s. Giving up." >&2
    exit 1
  fi
  echo "[pick_free_gpu] no free+unlocked GPU yet; retry in ${POLL_S}s" >&2
  sleep "$POLL_S"
  waited=$((waited + POLL_S))
done
