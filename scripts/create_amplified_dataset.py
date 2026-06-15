#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import (  # noqa: E402
    read_jsonl,
    records_to_dataframe,
    write_jsonl,
)
from medical_rag_thesis.splitting import assign_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the amplified Spanish dataset from SNS1064 and CasiMedicos."
    )
    parser.add_argument(
        "--sns-dir",
        default="data/processed/sns1064",
        help="Processed SNS1064 directory containing all/train/dev/test JSONL files.",
    )
    parser.add_argument(
        "--casimedicos",
        default="data/processed/casimedicos/all.jsonl",
        help="Normalized CasiMedicos JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/sns1064_casimedicos",
        help="Output directory for the amplified dataset.",
    )
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--dev-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. "
            "Create it first before building the amplified dataset."
        )


def load_sns_records(sns_dir: Path) -> list[dict[str, Any]]:
    all_path = sns_dir / "all.jsonl"
    require_path(all_path, "processed SNS1064 all.jsonl")
    records = read_jsonl(all_path)
    missing_split = [record.get("id") for record in records if not record.get("split")]
    if missing_split:
        raise ValueError(
            "SNS1064 records must already have train/dev/test splits. "
            f"Missing split for {len(missing_split)} records."
        )
    return records


def load_casimedicos_records(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    require_path(path, "normalized CasiMedicos JSONL")
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"No CasiMedicos records found in {path}")
    if all(record.get("split") for record in records):
        return records
    if len(records) < 3:
        return [{**record, "split": "train"} for record in records]
    return assign_splits(
        records,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        stratify_key="topic",
        seed=args.seed,
    )


def write_split_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_dir / "all.jsonl")
    records_to_dataframe(records).to_csv(output_dir / "all.csv", index=False)
    for split in ("train", "dev", "test"):
        split_records = [record for record in records if record.get("split") == split]
        write_jsonl(split_records, output_dir / f"{split}.jsonl")
        records_to_dataframe(split_records).to_csv(output_dir / f"{split}.csv", index=False)


def main() -> None:
    args = parse_args()
    sns_records = load_sns_records(Path(args.sns_dir))
    casimedicos_records = load_casimedicos_records(Path(args.casimedicos), args)
    records = [*sns_records, *casimedicos_records]
    output_dir = Path(args.output_dir)
    write_split_outputs(records, output_dir)

    summary = {
        "name": output_dir.name,
        "description": "Amplified Spanish QA dataset: SNS1064 plus CasiMedicos.",
        "num_records": len(records),
        "sources": dict(Counter(record.get("source", "") for record in records)),
        "splits": dict(Counter(record.get("split", "") for record in records)),
        "split_by_source": {
            source: dict(Counter(record.get("split", "") for record in records if record.get("source") == source))
            for source in sorted({record.get("source", "") for record in records})
        },
        "inputs": {
            "sns_dir": args.sns_dir,
            "casimedicos": args.casimedicos,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
