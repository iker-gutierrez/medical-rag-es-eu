#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import read_jsonl  # noqa: E402
from medical_rag_thesis.evaluation import evaluate_records, write_metrics  # noqa: E402
from medical_rag_thesis.run_logging import run_with_logs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated answers.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL from run_generation_experiment.py.")
    parser.add_argument("--output", required=True, help="Output metrics JSON.")
    parser.add_argument(
        "--references",
        help="Optional reference split JSONL. If omitted, the script tries predictions.meta.json input.",
    )
    parser.add_argument(
        "--semantic-model",
        default="intfloat/multilingual-e5-large",
        help="Sentence-transformers model for embedding cosine similarity. Use '' to skip.",
    )
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument(
        "--bertscore-model",
        default="",
        help="Optional BERTScore model, e.g. bert-base-multilingual-cased. Requires bert-score.",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-lang", default="es")
    parser.add_argument("--ragas", action="store_true", help="Request RAGAS metrics if configured.")
    return parser.parse_args()


def reference_path_from_metadata(predictions_path: Path) -> Optional[Path]:
    metadata_path = predictions_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_path = metadata.get("input")
    if not input_path:
        return None
    path = Path(input_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def run_metadata_from_predictions(predictions_path: Path) -> dict:
    metadata_path = predictions_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def enrich_records_with_references(
    records: list[dict],
    references_path: Optional[Path],
) -> list[dict]:
    if references_path is None or not references_path.exists():
        return records
    references = {record.get("id"): record for record in read_jsonl(references_path)}
    enriched = []
    for record in records:
        item = dict(record)
        reference = references.get(record.get("id"), {})
        if reference:
            item.setdefault("reference_short_answer", reference.get("short_answer", ""))
            item.setdefault("reference_evidence", reference.get("evidence", ""))
        enriched.append(item)
    return enriched


def run(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions)
    reference_path = Path(args.references) if args.references else reference_path_from_metadata(predictions_path)
    records = enrich_records_with_references(read_jsonl(predictions_path), reference_path)
    run_metadata = run_metadata_from_predictions(predictions_path)
    metrics = evaluate_records(
        records,
        run_metadata=run_metadata,
        semantic_model=args.semantic_model or None,
        semantic_batch_size=args.semantic_batch_size,
        bertscore_model=args.bertscore_model or None,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_lang=args.bertscore_lang,
        enable_ragas=args.ragas,
    )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))
    for warning in metrics.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    parsed_args = parse_args()
    output_path = Path(parsed_args.output)
    run_with_logs(
        output_path.with_suffix(".log"),
        output_path.with_suffix(".err"),
        lambda: run(parsed_args),
    )
