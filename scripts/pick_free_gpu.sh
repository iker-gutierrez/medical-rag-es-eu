#!/bin/bash
# Pin the current process to a genuinely-idle GPU, then run the given command,
# retrying with a fresh pick if vLLM still fails to reserve memory on it.
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
# The snapshot-then-claim is still a race against third parties: nvidia-smi sees a
# card as free, we claim it, but between that check and vLLM's actual reservation
# (after model load, tens of seconds to minutes later) another user's job -- not
# ours, so our lock does nothing for it -- can land on the same card. That is what
# failed task 6 of array 9575: "claimed GPU 8 (45460 MiB free)" followed minutes
# later by vLLM seeing only 11.11 GiB free. Since ConstrainDevices=no means nothing
# can fence the GPU to us at the OS level, the standard workaround on a shared,
# unfenced cluster is to make the launch self-healing: if the command fails, re-pick
# (fresh nvidia-smi snapshot, fresh claim, possibly a different card) and retry.
#
# Usage:  scripts/pick_free_gpu.sh <min_free_MiB> <command...>

set -uo pipefail   # no -e: a failed launch attempt must fall through to the retry loop

MIN_FREE_MIB="${1:-40000}"   # need ~40 GiB for a 0.90 reservation on a 46 GiB card
shift

MAX_WAIT_S=3600              # wait up to 1h for a card to free up
POLL_S=30
waited=0

MAX_LAUNCH_ATTEMPTS=3        # re-pick-and-retry this many times on a memory-race failure

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

attempt=0
while (( attempt < MAX_LAUNCH_ATTEMPTS )); do
  attempt=$((attempt + 1))

  # Consider cards ordered by free memory, and claim the first free-enough one whose
  # lock we can take. This inner loop only exits via `break 2` below (claimed a GPU)
  # or `exit 1` on timeout -- so reaching the code after it always means we have one.
  while true; do
    while IFS=',' read -r idx free; do
      idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
      [[ "$free" -ge "$MIN_FREE_MIB" ]] || continue
      if claim_gpu "$idx"; then
        export CUDA_VISIBLE_DEVICES="$idx"
        export GPU_CLAIM_LOCK="$LOCK_DIR/gpu${idx}.lock"
        echo "[pick_free_gpu] attempt ${attempt}/${MAX_LAUNCH_ATTEMPTS}: claimed GPU ${idx} (${free} MiB free >= ${MIN_FREE_MIB})"
        # Release the lock ~90s later: enough for vLLM to reserve its memory, after
        # which this card's occupancy is real and the next task will see it as busy.
        ( sleep 90; rmdir "$GPU_CLAIM_LOCK" 2>/dev/null || true ) &
        break 2
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

  # Run (not exec) so a launch failure -- e.g. a third party's process grabbed the
  # rest of this card between our snapshot and vLLM's actual reservation -- can be
  # caught and retried with a fresh pick, rather than killing the whole Slurm task.
  "$@"
  status=$?
  if (( status == 0 )); then
    exit 0
  fi
  echo "[pick_free_gpu] command failed (exit ${status}) on GPU ${CUDA_VISIBLE_DEVICES}; attempt ${attempt}/${MAX_LAUNCH_ATTEMPTS}" >&2
  if (( attempt < MAX_LAUNCH_ATTEMPTS )); then
    echo "[pick_free_gpu] retrying with a fresh GPU pick in ${POLL_S}s" >&2
    sleep "$POLL_S"
  fi
done

echo "[pick_free_gpu] command failed on all ${MAX_LAUNCH_ATTEMPTS} attempts. Giving up." >&2
exit 1
