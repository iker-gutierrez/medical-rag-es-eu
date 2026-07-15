#!/usr/bin/env python
"""True interspersed staged ablation: infer, evaluate, decide, infer, evaluate,
decide -- per model -- instead of running all 11 configs up front and patching
the dependent rows afterward.

For each model this enforces the intended dependency structure at *inference
time*, not after the fact:

  Stage A: baseline (row 0) + the six retrieval configs (rows 1-6).
           -> infer all 7 x 3 seeds -> evaluate -> compute MeanQ over the six
              retrieval configs -> pick the winner.
  Stage B: row 7 (few-shot, no RAG) -- independent of the winner, can run
           alongside stage A's rows -- and row 8 (few-shot + the stage-A
           MeanQ winner's retrieval settings, applied to row 8's config
           BEFORE it is run).
           -> infer -> evaluate -> compute MeanQ over rows 0-8 (used only to
              catch the degenerate case where a no-retrieval row would win the
              domain pool; see rewire_dependent_configs.py's docstring).
  Stage C: rows 9-10 (domain restriction: SNS-only index, CasiMedicos-only
           index), wired to the best RETRIEVING config from stage A/B (rows
           1-6 and 8; no-retrieval rows excluded -- a domain-restriction row
           must actually retrieve from its restricted corpus).
           -> infer -> evaluate.

Unlike scripts/rewire_dependent_configs.py (which patches an already-run grid
after the fact, only rewriting configs whose hardcoded base doesn't match),
this script never runs row 8, 9, or 10 with a wrong base in the first place --
the wiring decision happens before that row's config is submitted for
inference at all.

This is written for the NEXT ablation from scratch. It is not run automatically
against the current grid: rows 0-7 for every model already have valid, correct
predictions (verified independently -- see the seed/prompt audit earlier in
this project), and re-running them would throw away good data for no benefit.
Use scripts/rewire_dependent_configs.py + scripts/stage_ablation.py to bring an
already-run grid into a self-consistent state instead.

Usage:
  python scripts/staged_ablation_runner.py --models mistral7b qwen35_9b_no_think
  python scripts/staged_ablation_runner.py            # all 5 models
  python scripts/staged_ablation_runner.py --dry-run   # print the plan, run nothing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rewire_dependent_configs import (  # noqa: E402
    MODELS, base_retrieval_fields, apply_base,
)
from meanq import best_by_meanq  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
METRICS = ROOT / "reports" / "metrics"
SEEDS = [42, 43, 44]


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def submit_infer_array(configs: list[tuple[str, str]], label: str) -> str:
    """Submit a 1-GPU-per-task, %2-throttled Slurm array for the given
    (id_prefix, base) configs x 3 seeds. Mirrors stage_ablation.py's rerun
    array so the two orchestrators behave identically on the cluster."""
    tasks = [f"{prefix}_{base} {seed}" for prefix, base in configs for seed in SEEDS]
    task_file = ROOT / "experiments" / f"staged_{label}_tasks.txt"
    task_file.write_text("\n".join(tasks) + "\n")

    script = ROOT / "slurm" / f"staged_{label}.sh"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=staged-{label}
#SBATCH --array=0-{len(tasks) - 1}%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output={ROOT}/experiments/slurm_logs/staged_{label}_%A_%a.log
#SBATCH --error={ROOT}/experiments/slurm_logs/staged_{label}_%A_%a.err
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
    out = subprocess.run(["sbatch", "--parsable", str(script)], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    job = out.strip().split("_")[0]
    print(f"  [{label}] submitted array {job} ({len(tasks)} tasks, %2 throttle)", flush=True)
    return job


def wait_for_job(job: str) -> None:
    while True:
        q = subprocess.run(["squeue", "-j", job, "-h", "-o", "%t"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
        if not q:
            return
        time.sleep(30)


def evaluate(configs: list[tuple[str, str]], lang: str) -> None:
    for prefix, base in configs:
        for seed in SEEDS:
            run = f"{prefix}_{base}_seed{seed}"
            pred = RUNS / run / "predictions.jsonl"
            out = METRICS / f"{run}.json"
            if pred.exists() and not out.exists():
                print(f"  eval {run}", flush=True)
                sh([
                    sys.executable, "scripts/evaluate_predictions.py",
                    "--predictions", str(pred), "--output", str(out),
                    "--semantic-model", "",
                    "--bertscore-model", "bert-base-multilingual-cased",
                    "--bertscore-lang", lang,
                ])


def run_model(name: str, spec: dict, dry_run: bool) -> None:
    lang = spec["lang"]
    print(f"\n===== {name}: stage A (baseline + retrieval sweep) =====", flush=True)
    stage_a = [spec["baseline"]] + list(spec["retrieval"].values())
    if dry_run:
        print(f"  would infer+eval {len(stage_a)} configs: "
              f"{[b for _, b in stage_a]}")
    else:
        job = submit_infer_array(stage_a, f"{name}_A")
        wait_for_job(job)
        evaluate(stage_a, lang)

    win_r, scores_r = best_by_meanq(spec["retrieval"])
    if win_r is None:
        print(f"  {name}: no retrieval metrics yet (dry run or eval pending) -- stopping here")
        return
    print(f"  stage A MeanQ: " + ", ".join(f"{k}={v:.2f}" for k, v in
          sorted(scores_r.items(), key=lambda x: -x[1])))
    print(f"  stage A winner: {win_r}")

    print(f"\n===== {name}: stage B (row 7 few-shot no-RAG, row 8 few-shot+best-RAG) =====", flush=True)
    win_prefix, win_base = spec["retrieval"][win_r]
    base_fields = base_retrieval_fields(win_prefix, win_base)
    r8_prefix, r8_base = spec["row8"]
    apply_base(r8_prefix, r8_base, base_fields, apply=not dry_run)

    stage_b = [spec["fewshot_no_rag"], spec["row8"]]
    if dry_run:
        print(f"  would infer+eval {len(stage_b)} configs: {[b for _, b in stage_b]}")
    else:
        job = submit_infer_array(stage_b, f"{name}_B")
        wait_for_job(job)
        evaluate(stage_b, lang)

    print(f"\n===== {name}: stage C (domain restriction, rows 9-10) =====", flush=True)
    pre_domain = dict(spec["retrieval"])
    pre_domain["3-shot + best RAG"] = spec["row8"]
    win_pd, scores_pd = best_by_meanq(pre_domain)
    if win_pd is None:
        print(f"  {name}: stage B metrics not ready -- stopping here")
        return
    print(f"  domain base winner (retrieving configs only): {win_pd} "
          f"(MeanQ {scores_pd[win_pd]:.2f})")
    pd_prefix, pd_base = pre_domain[win_pd]
    domain_fields = base_retrieval_fields(pd_prefix, pd_base)
    domain_fields = {k: v for k, v in domain_fields.items() if k != "retrieval_index"}
    for corpus, (dprefix, dbase) in spec["domain"].items():
        apply_base(dprefix, dbase, domain_fields, apply=not dry_run)

    stage_c = list(spec["domain"].values())
    if dry_run:
        print(f"  would infer+eval {len(stage_c)} configs: {[b for _, b in stage_c]}")
    else:
        job = submit_infer_array(stage_c, f"{name}_C")
        wait_for_job(job)
        evaluate(stage_c, lang)

    print(f"\n{name}: staged run complete (11/11 configs).", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    help="Subset of models to run (default: all 5).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the stage plan; submit no Slurm jobs, write no configs.")
    args = ap.parse_args()

    for name in args.models:
        run_model(name, MODELS[name], args.dry_run)

    print("\nStaged runs complete. Regenerate tables with "
          "scripts/write_mixed_es_seed_summary.py / write_mixed_eu_seed_summary.py / "
          "write_result_tables.py, then choose the best config per language before "
          "the reasoning pipelines.")


if __name__ == "__main__":
    main()
