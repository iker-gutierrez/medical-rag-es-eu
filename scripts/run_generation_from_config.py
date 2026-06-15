#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generation experiment from a JSON config.")
    parser.add_argument("--config", required=True, help="Experiment config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Forward --dry-run to the generation script.")
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Forward --save-prompts to the generation script.",
    )
    return parser.parse_args()


def add_optional_arg(command: list[str], config: dict[str, Any], key: str, flag: str) -> None:
    value = config.get(key)
    if value is not None and value != "":
        command.extend([flag, str(value)])


def add_bool_arg(command: list[str], config: dict[str, Any], key: str, flag: str) -> None:
    if config.get(key):
        command.append(flag)


def build_command(config_path: Path, dry_run: bool, save_prompts: bool) -> list[str]:
    root = config_path.resolve().parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    command = [
        sys.executable,
        str(root / "scripts" / "run_generation_experiment.py"),
        "--input",
        config["input"],
        "--output",
        config["output"],
        "--experiment-name",
        config["experiment_name"],
    ]

    optional_args = {
        "model": "--model",
        "prompt_style": "--prompt-style",
        "language": "--language",
        "limit": "--limit",
        "seed": "--seed",
        "max_new_tokens": "--max-new-tokens",
        "temperature": "--temperature",
        "top_p": "--top-p",
        "dtype": "--dtype",
        "device_map": "--device-map",
        "few_shot_file": "--few-shot-file",
        "few_shot_k": "--few-shot-k",
        "few_shot_mode": "--few-shot-mode",
        "retrieval_index": "--retrieval-index",
        "retrieval_top_k": "--retrieval-top-k",
        "reranker_model": "--reranker-model",
        "reranker_top_k": "--reranker-top-k",
        "reranker_device": "--reranker-device",
        "feedback_max_new_tokens": "--feedback-max-new-tokens",
    }
    for key, flag in optional_args.items():
        add_optional_arg(command, config, key, flag)

    add_bool_arg(command, config, "think", "--think")
    add_bool_arg(command, config, "self_feedback", "--self-feedback")
    add_bool_arg(command, config, "trust_remote_code", "--trust-remote-code")
    if dry_run or config.get("dry_run"):
        command.append("--dry-run")
    if save_prompts or config.get("save_prompts"):
        command.append("--save-prompts")

    return command


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    command = build_command(config_path, dry_run=args.dry_run, save_prompts=args.save_prompts)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
