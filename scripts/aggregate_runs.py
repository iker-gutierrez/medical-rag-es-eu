#!/usr/bin/env python
"""Aggregate metrics from multiple runs of the same config (e.g. seeds 42/43/44).

Reads N metrics JSON files (one per run) and writes a single aggregated JSON
where every scalar metric is replaced by {"mean": ..., "std": ...}.  The output
has the same top-level shape as a single-run metrics file so summarize_metrics.py
can consume it via the --aggregated flag.

Usage:
    python scripts/aggregate_runs.py \
        --runs experiments/runs/123_foo/_run0/metrics.json \
                experiments/runs/123_foo/_run1/metrics.json \
                experiments/runs/123_foo/_run2/metrics.json \
        --output experiments/runs/123_foo/metrics_agg.json
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-run metrics into mean ± std.")
    parser.add_argument("--runs", nargs="+", required=True, help="Per-run metrics JSON files.")
    parser.add_argument("--output", required=True, help="Aggregated output metrics JSON.")
    return parser.parse_args()


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mu = sum(values) / n
    return sqrt(sum((x - mu) ** 2 for x in values) / (n - 1))


def _collect_scalars(payloads: list[Any]) -> dict[str, list[float]]:
    """Walk parallel dicts and collect lists of scalar values by dotted path."""
    result: dict[str, list[float]] = {}

    def _walk(nodes: list[Any], prefix: str) -> None:
        if not nodes or not isinstance(nodes[0], dict):
            return
        for key in nodes[0]:
            values = [node.get(key) for node in nodes if isinstance(node, dict)]
            path = f"{prefix}.{key}" if prefix else key
            scalars = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(scalars) == len(values) and scalars:
                result[path] = scalars
            elif all(isinstance(v, dict) for v in values if v is not None):
                _walk([v for v in values if isinstance(v, dict)], path)

    _walk(payloads, "")
    return result


def _aggregate_dict(nodes: list[Any]) -> Any:
    """Recursively aggregate a list of parallel dicts/values into mean ± std."""
    if not nodes:
        return None
    if all(isinstance(n, dict) for n in nodes):
        keys = nodes[0].keys()
        return {key: _aggregate_dict([n.get(key) for n in nodes]) for key in keys}
    if all(isinstance(n, list) for n in nodes):
        # e.g. warnings lists — just take the union
        seen: list[Any] = []
        for lst in nodes:
            for item in lst:
                if item not in seen:
                    seen.append(item)
        return seen
    numerics = [n for n in nodes if isinstance(n, (int, float)) and not isinstance(n, bool)]
    if len(numerics) == len(nodes) and numerics:
        mu = sum(numerics) / len(numerics)
        std = _stdev(numerics)
        return {"mean": mu, "std": std, "values": numerics}
    # non-numeric scalars (strings, bools, None) — return the first value
    return nodes[0]


def main() -> None:
    args = parse_args()
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.runs]

    summaries = [p.get("summary", {}) for p in payloads]
    agg_summary = _aggregate_dict(summaries)

    # Preserve per-run seeds/paths in metadata
    agg = {
        "summary": agg_summary,
        "aggregated": True,
        "num_runs": len(payloads),
        "run_files": [str(p) for p in args.runs],
        # rows are not aggregated — omit to keep file small
        "warnings": _aggregate_dict([p.get("warnings", []) for p in payloads]),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aggregated {len(payloads)} runs => {out}")

    # Print a quick summary of key metrics
    s = agg_summary
    for section in ("short_answer", "evidence", "overall"):
        sec = s.get(section, {})
        f1 = sec.get("token_overlap_f1", {})
        if isinstance(f1, dict):
            print(f"  {section} token_F1: {f1.get('mean', 0):.2f} ± {f1.get('std', 0):.2f}")


if __name__ == "__main__":
    main()
