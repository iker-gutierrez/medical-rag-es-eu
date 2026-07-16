#!/usr/bin/env python
"""Read the MA-RAG calibration runs and report the round-1 conflict distribution.

The calibration runs pin both thresholds above 1.0, so nothing ever settles and
every record reports the conflict its three candidates actually produced. Split
by source, that distribution is what the open-answer threshold should be set from.

Also reports structured-CoT format compliance: the first draft of that prompt made
the model emit its reasoning under the same headings as the answer fields and skip
the answer block entirely, which silently turned the whole chain of thought into
the "short answer". This checks the rewrite fixed it.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.reasoning import has_answer_label  # noqa: E402

RUNS = ROOT / "experiments" / "runs"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def report_conflict(tag: str) -> None:
    path = RUNS / f"calib_marag_{tag}" / "predictions.jsonl"
    if not path.exists():
        print(f"[{tag}] no calibration run at {path}")
        return
    records = [json.loads(line) for line in path.open()]
    by_source: dict[str, list[float]] = {}
    modes: dict[str, set[str]] = {}
    for record in records:
        log = record["reasoning_trace"]["round_log"]
        if not log:
            continue
        source = record.get("source") or "?"
        by_source.setdefault(source, []).append(float(log[0]["conflict"]))
        modes.setdefault(source, set()).add(log[0]["conflict_mode"])

    print(f"\n=== [{tag.upper()}] MA-RAG round-1 conflict distribution (n={len(records)}) ===")
    for source, values in sorted(by_source.items()):
        print(
            f"  {source:12} n={len(values):3}  mode={'/'.join(sorted(modes[source]))}\n"
            f"    min={min(values):.3f}  p25={quantile(values, .25):.3f}  "
            f"median={statistics.median(values):.3f}  p75={quantile(values, .75):.3f}  "
            f"max={max(values):.3f}  mean={statistics.mean(values):.3f}"
        )
        zeros = sum(1 for v in values if v == 0.0)
        print(f"    unanimous (conflict==0): {zeros}/{len(values)}")
        for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
            triggered = sum(1 for v in values if v > threshold)
            print(
                f"      threshold {threshold:.2f} -> {triggered:3}/{len(values)} "
                f"({100 * triggered / len(values):5.1f}%) would iterate"
            )


def report_format(tag: str) -> None:
    path = RUNS / f"calib_scot_{tag}" / "predictions.jsonl"
    if not path.exists():
        print(f"\n[{tag}] no structured-CoT calibration run at {path}")
        return
    records = [json.loads(line) for line in path.open()]
    labelled = sum(1 for r in records if has_answer_label(r["prediction_text"]))
    evidence = sum(1 for r in records if (r["parsed_prediction"] or {}).get("evidence"))
    print(f"\n=== [{tag.upper()}] structured-CoT format compliance (n={len(records)}) ===")
    print(f"  emitted the answer label : {labelled}/{len(records)}")
    print(f"  parsed an evidence field : {evidence}/{len(records)}")
    bad = [r["id"] for r in records if not has_answer_label(r["prediction_text"])]
    if bad:
        print(f"  MISSING answer label    : {bad}")
        example = next(r for r in records if r["id"] == bad[0])
        print(f"  example output (300 ch) : {example['prediction_text'][:300]!r}")


def main() -> None:
    for tag in ("eu", "es", "eu_latxa_topk1"):
        report_conflict(tag)
        report_format(tag)


if __name__ == "__main__":
    main()
