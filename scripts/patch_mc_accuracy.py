#!/usr/bin/env python
"""Patch mc_accuracy into existing *_casimedicos.json metric files in place.

Why this exists: the CasiMedicos metric subsets for the Qwen runs (and Mistral
seeds 43/44) were produced by an evaluator that predated the mc_accuracy metric,
so mc_accuracy is absent/None there. That made MeanQ = mean(ROUGE-L, BERT-F1,
MC-acc) fall back to two components for those configs, and MC-acc was only
seed-42-averaged for Mistral -- i.e. the config selection was not consistently
computed. The predictions already exist, so mc_accuracy is a pure recompute with
no model loading and no GPU.

This reuses the evaluator's OWN functions (parsed_prediction_sections,
mc_accuracy) and the eval CLI's reference join (enrich_records_with_references)
so the patched value matches exactly what a full re-eval would produce -- it just
skips the expensive ROUGE-L/BERT-F1 recompute and leaves those untouched.

It writes mc_accuracy for BOTH the initial (before_feedback) and final
(after_feedback / top-level overall) predictions, into the same JSON structure
the evaluator uses:
    summary.short_answer.mc_accuracy
    summary.overall.mc_accuracy
    summary.before_feedback.{short_answer,overall}.mc_accuracy
    summary.after_feedback.{short_answer,overall}.mc_accuracy   (== top-level)

Run:  python scripts/patch_mc_accuracy.py            # patch all that need it
      python scripts/patch_mc_accuracy.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag_thesis.evaluation import (  # noqa: E402
    mc_accuracy,
    parsed_prediction_sections,
    percent,
)
from evaluate_predictions import (  # noqa: E402
    enrich_records_with_references,
    read_jsonl,
    reference_path_from_metadata,
)

METRICS = ROOT / "reports" / "metrics"
RUNS = ROOT / "experiments" / "runs"


def predictions_path_for(metric_name: str) -> Path | None:
    """`<run>_casimedicos.json` -> experiments/runs/<run>/predictions.jsonl.

    The run dir is the metric file name minus the `_casimedicos` suffix.
    """
    run = metric_name[: -len("_casimedicos.json")]
    p = RUNS / run / "predictions.jsonl"
    return p if p.exists() else None


def mc_for_prediction(records: list[dict], *, initial: bool) -> float | None:
    """Mean mc_accuracy over the multiple-choice records, using the same
    prediction section the evaluator scores.

    initial=True  -> parsed_initial_prediction (before self-feedback)
    initial=False -> parsed_prediction        (final / after)
    """
    kwargs = (
        dict(parsed_key="parsed_initial_prediction", text_key="initial_prediction_text")
        if initial
        else {}
    )
    vals: list[float] = []
    for record in records:
        options = record.get("options") or {}
        if not options:
            continue
        sections = parsed_prediction_sections(record, **kwargs)
        prediction = sections.get("short_answer", "")
        v = mc_accuracy(prediction, options, record.get("correct_option"))
        if v is not None:
            vals.append(v)
    return percent(statistics.mean(vals)) if vals else None


def patch_file(metric_path: Path, *, dry_run: bool) -> str:
    preds = predictions_path_for(metric_path.name)
    if preds is None:
        return "NO-PREDICTIONS"

    ref_path = reference_path_from_metadata(preds)
    records = enrich_records_with_references(read_jsonl(preds), ref_path)
    # CasiMedicos subset = records that carry options (multiple-choice).
    mc_records = [r for r in records if r.get("options")]
    if not mc_records:
        return "NO-MC-RECORDS"

    mc_final = mc_for_prediction(mc_records, initial=False)
    mc_initial = mc_for_prediction(mc_records, initial=True)
    if mc_initial is None:
        # runs without self-feedback: initial == final
        mc_initial = mc_final

    data = json.loads(metric_path.read_text())
    summary = data.get("summary") or {}

    def set_mc(block: dict | None, value: float | None) -> None:
        if block is None:
            return
        for sec in ("short_answer", "overall"):
            if isinstance(block.get(sec), dict):
                block[sec]["mc_accuracy"] = value

    # top-level == final prediction (after feedback)
    set_mc(summary, mc_final)
    set_mc(summary.get("after_feedback"), mc_final)
    set_mc(summary.get("before_feedback"), mc_initial)

    if dry_run:
        return f"WOULD-SET initial={mc_initial} final={mc_final}"

    metric_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return f"SET initial={mc_initial} final={mc_final}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Only files that the tables actually read: <id>_<base>_seed{42,43,44}_casimedicos.json.
    import re
    targets = [
        p for p in sorted(METRICS.glob("*_casimedicos.json"))
        if re.search(r"_seed(42|43|44)_casimedicos\.json$", p.name)
        and '"mc_accuracy"' not in p.read_text()
    ]
    print(f"{len(targets)} metric files need mc_accuracy\n")
    counts: dict[str, int] = {}
    for p in targets:
        status = patch_file(p, dry_run=args.dry_run)
        key = status.split()[0]
        counts[key] = counts.get(key, 0) + 1
        print(f"  {status:40} {p.name}")
    print("\nsummary:", counts)


if __name__ == "__main__":
    main()
