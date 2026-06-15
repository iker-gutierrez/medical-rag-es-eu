#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import (  # noqa: E402
    clean_text,
    compact_record,
    ensure_id,
    find_column,
    read_table,
    records_to_dataframe,
    write_jsonl,
)
from medical_rag_thesis.splitting import assign_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and split the SNS1064 dataset.")
    parser.add_argument("--input", required=True, help="Raw SNS1064 file: csv, tsv, jsonl, json, xlsx.")
    parser.add_argument("--output-dir", required=True, help="Directory for all/train/dev/test files.")
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--dev-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="SNS1064")
    return parser.parse_args()


def normalize_sns1064(input_path: str, source: str) -> list[dict[str, str]]:
    df = read_table(input_path)
    id_col = find_column(df, ["id", "sample_id", "question_id"], required=False)
    guidebook_col = find_column(df, ["guidebook", "guide_book", "guia", "guía"], required=False)
    topic_col = find_column(df, ["Topic", "topic", "tema"], required=False)
    question_col = find_column(df, ["Question", "question", "pregunta"])
    subquestion_col = find_column(
        df,
        ["subquestion", "sub_question", "subpregunta", "sub pregunta"],
        required=False,
    )
    short_answer_col = find_column(
        df,
        ["Short answer", "short_answer", "answer", "judgement", "juicio", "respuesta corta", "respuesta"],
    )
    evidence_col = find_column(df, ["evidence"], required=False)
    additional_col = find_column(df, ["considerations"], required=False)
    records = []
    for index, row in df.iterrows():
        question = clean_text(row[question_col])
        subquestion = clean_text(row[subquestion_col]) if subquestion_col else ""
        evidence_parts = []
        evidence = clean_text(row[evidence_col]) if evidence_col else ""
        additional = clean_text(row[additional_col]) if additional_col else ""
        if evidence:
            evidence_parts.append(evidence)
        if additional:
            evidence_parts.append(additional)
        record = compact_record(
            {
                "id": ensure_id("sns1064", int(index), row[id_col] if id_col else None),
                "source": source,
                "guidebook": clean_text(row[guidebook_col]) if guidebook_col else "",
                "topic": clean_text(row[topic_col]) if topic_col else "",
                "question": question,
                "subquestion": subquestion,
                "short_answer": clean_text(row[short_answer_col]),
                "evidence": "\n\n".join(evidence_parts),
            }
        )
        if record.get("question") and record.get("short_answer"):
            records.append(record)
    if not records:
        raise ValueError("No usable SNS1064 rows found. Check column names and empty values.")
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = normalize_sns1064(args.input, args.source)
    records = assign_splits(
        records,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    write_jsonl(records, output_dir / "all.jsonl")
    records_to_dataframe(records).to_csv(output_dir / "all.csv", index=False)

    summary = {
        "input": str(args.input),
        "num_records": len(records),
        "splits": dict(Counter(record["split"] for record in records)),
        "topics": dict(Counter(record.get("topic", "") for record in records if record.get("topic"))),
        "seed": args.seed,
    }
    for split in ["train", "dev", "test"]:
        split_records = [record for record in records if record["split"] == split]
        write_jsonl(split_records, output_dir / f"{split}.jsonl")
        records_to_dataframe(split_records).to_csv(output_dir / f"{split}.csv", index=False)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
