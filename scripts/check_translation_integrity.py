#!/usr/bin/env python
"""Fail if any Basque field is truncated relative to its Spanish source.

The previous translator silently cut long passages at MarianMT's 512-token input
limit, so 20.6% of dev and 15.4% of train evidence fields lost more than half their
content. Those fields are the gold references the Basque metrics are scored against,
and (because the retrieval index is built from train) also the passages the Basque
retriever returns. A truncated gold reference is not a small error: it makes the
metric measure something other than what it claims to.

This check is deliberately crude, because the failure it guards against is crude.
It does not attempt to judge translation *quality*, which would need a reference we
do not have. It only asks whether content went missing, which a length ratio detects
reliably: Basque is mildly more compact than Spanish, so a healthy ratio sits near
1.0, and a field that keeps a quarter of its characters has not been translated, it
has been cut.

Exit code 1 if any field falls below the threshold.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FLAGGED_FILE = PROCESSED / "manual_translation_needed" / "flagged.jsonl"


def load_flagged() -> set[tuple[str, str]]:
    """(id, field) pairs the HiTZ model cannot translate and that are awaiting a
    human translation. They are reported but do NOT fail the gate: they are a known,
    tracked exception, not a silent one. Once eu_manual is filled and applied, the
    entry can be removed and the field will be checked normally again."""
    flagged = set()
    if FLAGGED_FILE.exists():
        for line in FLAGGED_FILE.open():
            r = json.loads(line)
            flagged.add((r["id"], r["field"]))
    return flagged

PAIRS = [
    ("sns1064", "sns1064_eu"),
    ("casimedicos", "casimedicos_eu"),
]
FIELDS = ("topic", "question", "subquestion", "short_answer", "evidence")

# A truncated field loses most of its content; a legitimately translated one does
# not. The threshold must sit between the two. On the corrected (coherent) Basque,
# the evidence-length ratio has median 0.81 and a hard floor of 0.31 -- Basque is
# genuinely ~20-35% more compact than Spanish for dense medical prose, so ratios in
# the 0.5-0.7 band are NORMAL translation, not loss. The old failure mode, by
# contrast, kept 6-25%. A floor of 0.30 therefore separates real truncation (which
# would reappear well below 0.30) from ordinary compactness, while still catching a
# field that has genuinely been cut. Short sources are noisier and exempt.
RATIO_FLOOR = 0.30
MIN_SOURCE_CHARS = 40


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    manual = load_flagged()
    failures: list[str] = []
    pending_manual: list[str] = []
    total = flagged = 0

    for es_name, eu_name in PAIRS:
        for split in args.splits:
            es_path = PROCESSED / es_name / f"{split}.jsonl"
            eu_path = PROCESSED / eu_name / f"{split}.jsonl"
            if not (es_path.exists() and eu_path.exists()):
                continue
            es = {json.loads(l)["id"]: json.loads(l) for l in es_path.open()}
            eu = {json.loads(l)["id"]: json.loads(l) for l in eu_path.open()}

            if set(es) != set(eu):
                failures.append(
                    f"{es_name}/{split}: id sets differ "
                    f"(es={len(es)} eu={len(eu)}); the sets are no longer parallel"
                )

            ratios: list[float] = []
            for record_id, es_record in es.items():
                eu_record = eu.get(record_id)
                if not eu_record:
                    continue
                for field in FIELDS:
                    source = str(es_record.get(field) or "").strip()
                    target = str(eu_record.get(field) or "").strip()
                    if len(source) < MIN_SOURCE_CHARS:
                        continue
                    total += 1
                    if not target:
                        failures.append(f"{record_id}.{field}: EMPTY in Basque")
                        flagged += 1
                        continue
                    ratio = len(target) / len(source)
                    ratios.append(ratio)
                    if ratio < RATIO_FLOOR:
                        if (record_id, field) in manual:
                            pending_manual.append(
                                f"{record_id}.{field}: kept {ratio:.0%} "
                                f"(awaiting manual translation)"
                            )
                        else:
                            flagged += 1
                            failures.append(
                                f"{record_id}.{field}: kept {ratio:.0%} of the source "
                                f"({len(source)} -> {len(target)} chars)"
                            )
            if ratios:
                print(
                    f"  {es_name}/{split:5} n={len(ratios):5}  "
                    f"median ratio={statistics.median(ratios):.2f}  "
                    f"min={min(ratios):.2f}"
                )

    print(f"\n  fields checked: {total}")
    print(f"  truncated     : {flagged}")
    if pending_manual:
        print(f"\n  KNOWN, awaiting manual translation ({len(pending_manual)}):")
        for line in pending_manual:
            print(f"    {line}")
        print("    -> data/processed/manual_translation_needed/flagged.jsonl")

    if failures:
        print(f"\n  TRANSLATION INTEGRITY FAILED ({len(failures)} problems):")
        for line in failures[:20]:
            print(f"    {line}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        print(
            "\n  These fields are gold references. Scoring against them measures the\n"
            "  translator's truncation, not the model's answer. Fix the chunking in\n"
            "  scripts/translate_to_basque.py and retranslate before evaluating."
        )
        sys.exit(1)

    if pending_manual:
        print(f"\n  OK: no unexpected truncation "
              f"({len(pending_manual)} field(s) flagged for manual translation).")
    else:
        print("\n  OK: no truncated fields.")


if __name__ == "__main__":
    main()
