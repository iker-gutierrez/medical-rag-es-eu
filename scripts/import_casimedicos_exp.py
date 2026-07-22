#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import compact_record, records_to_dataframe, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and normalize the Spanish HiTZ/casimedicos-exp dataset."
    )
    parser.add_argument("--dataset", default="HiTZ/casimedicos-exp")
    parser.add_argument("--config", default="es")
    parser.add_argument("--raw-dir", default="data/raw/casimedicos_exp")
    parser.add_argument("--output-dir", default="data/processed/casimedicos")
    return parser.parse_args()


def split_name(name: str) -> str:
    return "dev" if name == "validation" else name


def extract_evidence(row: dict[str, Any]) -> str:
    """Use full_answer (the expert explanation) as retrieval evidence."""
    return (row.get("full_answer") or "").strip()


def normalize_record(row: dict[str, Any], split: str) -> Optional[dict[str, Any]]:
    question = (row.get("full_question") or "").strip()
    correct_option = str(row.get("correct_option", ""))
    options = {str(k): v for k, v in (row.get("options") or {}).items() if v}
    short_answer = options.get(correct_option, correct_option)
    evidence = extract_evidence(row)

    if not question or not short_answer:
        return None

    return compact_record(
        {
            "id": f"casimedicos_{row.get('id')}",
            "source": "CasiMedicos",
            "split": split,
            "year": row.get("year"),
            "topic": (row.get("type") or "").strip(),
            "question": question,
            "options": options,
            "correct_option": correct_option,
            "short_answer": short_answer,
            "evidence": evidence,
        }
    )


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_dir / "all.jsonl")
    records_to_dataframe(records).to_csv(output_dir / "all.csv", index=False)
    for split in ("train", "dev", "test"):
        split_records = [r for r in records if r.get("split") == split]
        write_jsonl(split_records, output_dir / f"{split}.jsonl")
        records_to_dataframe(split_records).to_csv(output_dir / f"{split}.csv", index=False)


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    dataset = load_dataset(args.dataset, args.config)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    raw_records: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []

    for hf_split, split_dataset in dataset.items():
        split = split_name(hf_split)
        split_raw = []
        split_normalized = []
        for row in split_dataset:
            row_dict = dict(row)
            split_raw.append({**row_dict, "split": split})
            normalized = normalize_record(row_dict, split)
            if normalized:
                split_normalized.append(normalized)
        raw_records.extend(split_raw)
        normalized_records.extend(split_normalized)

    write_jsonl(raw_records, raw_dir / "all.jsonl")
    records_to_dataframe(raw_records).to_csv(raw_dir / "all.csv", index=False)
    for split in ("train", "dev", "test"):
        split_raw = [r for r in raw_records if r.get("split") == split]
        write_jsonl(split_raw, raw_dir / f"{split}.jsonl")
        records_to_dataframe(split_raw).to_csv(raw_dir / f"{split}.csv", index=False)

    write_outputs(normalized_records, output_dir)

    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "num_raw_records": len(raw_records),
        "num_normalized_records": len(normalized_records),
        "splits": dict(Counter(r["split"] for r in normalized_records)),
        "sources": dict(Counter(r["source"] for r in normalized_records)),
        "note": "Splits are as published by HiTZ authors on HuggingFace (not resampled).",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
