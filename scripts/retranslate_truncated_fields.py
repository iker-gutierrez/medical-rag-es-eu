#!/usr/bin/env python
"""Retranslate exactly the fields flagged as truncated by
check_translation_integrity.py, using the fixed paragraph+sentence chunking
in translate_to_basque.py (sec:translation-artefact). Leaves every other
field untouched -- this is a targeted patch, not a full re-translation.

Usage:
  python scripts/retranslate_truncated_fields.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PROCESSED = ROOT / "data" / "processed"

from translate_to_basque import load_model, translate_text  # noqa: E402

PAIRS = [
    ("sns1064", "sns1064_eu"),
    ("casimedicos", "casimedicos_eu"),
]
FIELDS = ("topic", "question", "subquestion", "short_answer", "evidence")
RATIO_FLOOR = 0.30
MIN_SOURCE_CHARS = 40


def find_truncated() -> list[tuple[str, str, str, str, str]]:
    """Returns (es_name, eu_name, split, record_id, field) for every field
    below RATIO_FLOOR, matching check_translation_integrity.py's own logic
    exactly so this script retranslates precisely what that check flags."""
    hits = []
    for es_name, eu_name in PAIRS:
        for split in ("train", "dev", "test"):
            es_path = PROCESSED / es_name / f"{split}.jsonl"
            eu_path = PROCESSED / eu_name / f"{split}.jsonl"
            if not (es_path.exists() and eu_path.exists()):
                continue
            es = {json.loads(l)["id"]: json.loads(l) for l in es_path.open()}
            eu = {json.loads(l)["id"]: json.loads(l) for l in eu_path.open()}
            for record_id, es_record in es.items():
                eu_record = eu.get(record_id)
                if not eu_record:
                    continue
                for field in FIELDS:
                    source = str(es_record.get(field) or "").strip()
                    target = str(eu_record.get(field) or "").strip()
                    if len(source) < MIN_SOURCE_CHARS:
                        continue
                    if not target or len(target) / len(source) < RATIO_FLOOR:
                        hits.append((es_name, eu_name, split, record_id, field))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="HiTZ/medical_es-eu")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hits = find_truncated()
    print(f"{len(hits)} truncated fields found")
    by_file: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for es_name, eu_name, split, record_id, field in hits:
        by_file.setdefault((eu_name, split), []).append((record_id, field))
    for (eu_name, split), items in sorted(by_file.items()):
        print(f"  {eu_name}/{split}: {len(items)} fields")

    if args.dry_run:
        return

    tokenizer, model = load_model(args.model, args.device)

    es_name_for_eu = {eu: es for es, eu in PAIRS}
    for (eu_name, split), items in sorted(by_file.items()):
        es_name = es_name_for_eu[eu_name]
        es_path = PROCESSED / es_name / f"{split}.jsonl"
        eu_path = PROCESSED / eu_name / f"{split}.jsonl"
        es_records = {json.loads(l)["id"]: json.loads(l) for l in es_path.open()}
        eu_lines = eu_path.read_text(encoding="utf-8").splitlines()
        eu_records = [json.loads(l) for l in eu_lines if l.strip()]
        eu_by_id = {r["id"]: r for r in eu_records}

        print(f"\n{eu_name}/{split}: retranslating {len(items)} fields")
        for i, (record_id, field) in enumerate(items, start=1):
            source_text = str(es_records[record_id].get(field) or "").strip()
            new_translation = translate_text(source_text, tokenizer, model, args.device)
            old = eu_by_id[record_id].get(field, "")
            eu_by_id[record_id][field] = new_translation
            print(
                f"  [{i}/{len(items)}] {record_id}.{field}: "
                f"{len(old)} -> {len(new_translation)} chars "
                f"(source {len(source_text)} chars)"
            )

        eu_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in eu_records) + "\n",
            encoding="utf-8",
        )
        print(f"  saved {eu_path}")

    print("\nAll done. Rebuilding the combined sns1064_casimedicos_eu files and the "
          "retrieval index is a separate step -- see slurm/translate_eu.sh's combine "
          "step and scripts/build_retrieval_index.py.")


if __name__ == "__main__":
    main()
