#!/usr/bin/env python
"""Fail loudly if any run in the seeded ablation produced a truncated record.

The archived (pre-seed-fix) runs of these same configs truncated 0 / 20790
records, and the configs now additionally carry repetition_detection plus
max_truncation_retries=3. So the expected result here is zero. This script exists
so that a regression shows up as a failed check rather than as a quietly degraded
row in a results table -- a truncated generation still gets scored, it just gets
scored on a half-written answer.

Run after generation, before evaluation. Exit code 1 if anything truncated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"

INCOMPLETE = ("length", "repetition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file",
        default="/tmp/claude-1034/-home-igutierrez134/033c132e-359b-4152-9f88-238f66c6f423/scratchpad/new_configs.txt",
        help="File of config stems; each is checked at seeds 42/43/44.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    stems = Path(args.runs_file).read_text().split()
    total_records = truncated_records = 0
    missing: list[str] = []
    offenders: list[tuple[str, int, int]] = []

    for stem in stems:
        for seed in args.seeds:
            run = f"{stem}_seed{seed}"
            path = RUNS / run / "predictions.jsonl"
            if not path.exists():
                missing.append(run)
                continue
            bad = seen = 0
            for line in path.open():
                record = json.loads(line)
                seen += 1
                truncation = record.get("truncation") or {}
                if (
                    truncation.get("initial_finish_reason") in INCOMPLETE
                    or truncation.get("feedback_finish_reason") in INCOMPLETE
                ):
                    bad += 1
            total_records += seen
            truncated_records += bad
            if bad:
                offenders.append((run, bad, seen))

    print(f"  runs checked   : {len(stems) * len(args.seeds) - len(missing)}")
    print(f"  records checked: {total_records}")
    print(f"  truncated      : {truncated_records}")

    if missing:
        print(f"\n  NOT YET GENERATED ({len(missing)}):")
        for run in missing[:10]:
            print(f"    {run}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if offenders:
        print(f"\n  TRUNCATION DETECTED in {len(offenders)} run(s):")
        for run, bad, seen in sorted(offenders, key=lambda x: -x[1]):
            print(f"    {bad:4}/{seen}  {run}")
        print(
            "\n  These rows would be scored on half-written answers. Raise max_new_tokens\n"
            "  (or thinking_token_budget) for the affected configs and regenerate them."
        )
        sys.exit(1)

    if not missing:
        print("\n  OK: no truncated records.")


if __name__ == "__main__":
    main()
