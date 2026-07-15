#!/usr/bin/env python
"""Staged ablation driver: evaluate, pick MeanQ-best, wire dependents, re-run, repeat.

The ablation grid is staged: the few-shot-plus-RAG row is meant to use the best
retrieval configuration, and the domain-restriction rows the best configuration so
far. Because every config already ran once with a hardcoded rerank-top-5 base, the
staging here takes the form of a correction loop rather than a from-scratch
sequence:

  1. Evaluate everything that has predictions but no metrics.
  2. Compute MeanQ per stage and rewrite any dependent config whose base does not
     match the MeanQ winner (scripts/rewire_dependent_configs.py).
  3. If any config changed, its old predictions are now stale: re-run exactly those
     configs (all three seeds) and loop back to step 1.
  4. When a full pass changes nothing, the grid is self-consistent: regenerate the
     result tables and stop.

This runs automatically for the ablation. It deliberately stops before the reasoning
pipelines, which require a manual choice of the best configuration per language.

It orchestrates GPU work by submitting Slurm jobs and waiting on them, so it must run
somewhere it can call sbatch (a login node or a CPU Slurm job); it does no GPU work
itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONFIG_LIST = Path(
    "/tmp/claude-1034/-home-igutierrez134/033c132e-359b-4152-9f88-238f66c6f423/scratchpad/new_configs.txt"
)
METRICS = ROOT / "reports" / "metrics"
RUNS = ROOT / "experiments" / "runs"
SEEDS = [42, 43, 44]
MAX_PASSES = 4  # row8 then domain then converge; a safety ceiling


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout


def evaluate_all() -> None:
    """Evaluate every run that has predictions but no metric file yet (CPU+1 GPU,
    cosine off -- only the shown metrics)."""
    configs = CONFIG_LIST.read_text().split()
    for cfg in configs:
        lang = "eu" if cfg.endswith("_eu_dev") else "es"
        for seed in SEEDS:
            run = f"{cfg}_seed{seed}"
            pred = RUNS / run / "predictions.jsonl"
            out = METRICS / f"{run}.json"
            if pred.exists() and not out.exists():
                print(f"  eval {run}", flush=True)
                subprocess.run([
                    sys.executable, "scripts/evaluate_predictions.py",
                    "--predictions", str(pred), "--output", str(out),
                    "--semantic-model", "",
                    "--bertscore-model", "bert-base-multilingual-cased",
                    "--bertscore-lang", lang,
                ], cwd=ROOT, check=True)


def rewire() -> list[str]:
    """Apply MeanQ rewiring; return the list of configs whose base changed."""
    changed_file = ROOT / "experiments" / "meanq_changed.txt"
    if changed_file.exists():
        changed_file.unlink()
    subprocess.run([
        sys.executable, "scripts/rewire_dependent_configs.py",
        "--apply", "--out", str(changed_file),
    ], cwd=ROOT, check=True)
    if not changed_file.exists():
        return []
    return [c for c in changed_file.read_text().split() if c]


def model_label(cfg: str) -> str:
    """Short model tag for job names, so squeue distinguishes which model a rerun
    array belongs to instead of every array showing the same generic name."""
    if "latxa" in cfg:
        return "latxa"
    if "llama31_8b" in cfg:
        return "llama"
    if "mistral7b" in cfg:
        return "mistral"
    if "qwen35_9b" in cfg:
        return "qwen-think" if "_think_" in cfg and "no_think" not in cfg else "qwen-notk"
    return "model"


def rerun_configs(configs: list[str], wait: bool) -> None:
    """Re-run the changed dependent configs (all seeds), discarding their stale
    outputs first so evaluate_all picks the new ones up."""
    if not configs:
        return
    tasks = []
    for cfg in configs:
        for seed in SEEDS:
            run_dir = RUNS / f"{cfg}_seed{seed}"
            metric = METRICS / f"{cfg}_seed{seed}.json"
            # Stale outputs from the wrong base must go, or they would be reused.
            for p in (run_dir / "predictions.jsonl", metric):
                if p.exists():
                    p.unlink()
            for extra in METRICS.glob(f"{cfg}_seed{seed}_*.json"):
                extra.unlink()
            tasks.append(f"{cfg} {seed}")

    task_file = ROOT / "experiments" / "meanq_rerun_tasks.txt"
    print(f"  re-running {len(tasks)} dependent runs (changed base):", flush=True)
    for t in tasks:
        print(f"    {t}")
    task_file.write_text("\n".join(tasks) + "\n")

    # Job name carries the model(s) in this batch (e.g. "abl-qwen-think") so squeue
    # shows which rerun is which -- previously every batch was named "abl-meanq"
    # regardless of model, and the only way to tell them apart was to cross-reference
    # job IDs against this script's own task-list output.
    labels = sorted(set(model_label(cfg) for cfg in configs))
    label = "-".join(labels)[:20]
    job = submit_rerun_array(len(tasks), task_file, label)
    if wait:
        wait_for_job(job)


def submit_rerun_array(n: int, task_file: Path, label: str = "meanq") -> str:
    script = ROOT / "slurm" / f"ablation_{label}_rerun.sh"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=abl-{label}
#SBATCH --array=0-{n - 1}%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output={ROOT}/experiments/slurm_logs/abl_{label}_%A_%a.log
#SBATCH --error={ROOT}/experiments/slurm_logs/abl_{label}_%A_%a.err
#SBATCH --chdir={ROOT}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus
set -euo pipefail
mapfile -t TASKS < {task_file}
ENTRY="${{TASKS[$SLURM_ARRAY_TASK_ID]}}"
CONFIG="${{ENTRY%% *}}"; SEED="${{ENTRY##* }}"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \\
  --config "configs/experiments/${{CONFIG}}.json" --seed "${{SEED}}"
""")
    script.chmod(0o755)
    jid = sh(["sbatch", "--parsable", str(script)]).strip()
    print(f"  submitted rerun array {jid}", flush=True)
    return jid.split("_")[0]


def wait_for_job(job: str) -> None:
    while True:
        q = sh(["squeue", "-j", job, "-h", "-o", "%t"]).split()
        if not q:
            return
        time.sleep(30)


def regenerate_tables() -> None:
    for script in ("write_mixed_es_seed_summary.py", "write_mixed_eu_seed_summary.py",
                   "write_result_tables.py"):
        subprocess.run([sys.executable, f"scripts/{script}"], cwd=ROOT, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true",
                    help="Submit reruns but don't block (for interactive use).")
    args = ap.parse_args()

    for pass_no in range(1, MAX_PASSES + 1):
        print(f"\n===== staged pass {pass_no} =====", flush=True)
        evaluate_all()
        changed = rewire()
        if not changed:
            print("  grid is self-consistent: no dependent config changed.", flush=True)
            break
        print(f"  {len(changed)} configs re-wired to their MeanQ base; re-running.", flush=True)
        rerun_configs(changed, wait=not args.no_wait)
    else:
        print("  reached MAX_PASSES; stopping (check for oscillation).", flush=True)

    print("\n===== regenerating result tables =====", flush=True)
    regenerate_tables()
    print("\nStaged ablation complete. STOP: choose the best config per language before "
          "the reasoning pipelines.", flush=True)


if __name__ == "__main__":
    main()
