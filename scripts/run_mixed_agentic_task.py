#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys

from mixed_agentic_tasks import ROOT, predictions_path, select_best, task_by_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one mixed-agentic dev task.")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--source-answer-key", default="initial", choices=("initial", "final"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = task_by_index(args.task_index)
    baseline_run = select_best(task.baseline_pool)
    candidate_run = select_best(task.candidate_pool)
    output = predictions_path(task.run_id)

    print(f"Mixed agentic task index: {args.task_index}", flush=True)
    print(f"RUN_ID={task.run_id}", flush=True)
    print(f"DEV_SET={task.dev_set}", flush=True)
    print(f"LANGUAGE={task.language}", flush=True)
    print(f"BASELINE_RUN={baseline_run}", flush=True)
    print(f"CANDIDATE_RUN={candidate_run}", flush=True)
    print(f"JUDGE_MODEL={task.judge_model}", flush=True)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_agentic_reasoner.py"),
        "--experiment-name",
        task.run_id,
        "--baseline-predictions",
        str(predictions_path(baseline_run).relative_to(ROOT)),
        "--candidate-predictions",
        str(predictions_path(candidate_run).relative_to(ROOT)),
        "--output",
        str(output.relative_to(ROOT)),
        "--model",
        task.judge_model,
        "--language",
        task.language,
        "--source-answer-key",
        args.source_answer_key,
        "--max-new-tokens",
        str(task.max_new_tokens),
    ]
    if task.trust_remote_code:
        command.append("--trust-remote-code")
    if args.dry_run:
        command.append("--dry-run")
    if args.save_prompts:
        command.append("--save-prompts")

    subprocess.run(command, cwd=ROOT, check=True)

    eval_command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_predictions.py"),
        "--predictions",
        str(output.relative_to(ROOT)),
        "--output",
        str((ROOT / "reports" / "metrics" / f"{task.run_id}.json").relative_to(ROOT)),
        "--semantic-model",
        "intfloat/multilingual-e5-large",
        "--bertscore-model",
        "bert-base-multilingual-cased",
        "--bertscore-lang",
        task.language,
    ]
    subprocess.run(eval_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
