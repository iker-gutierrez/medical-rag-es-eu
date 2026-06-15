#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.retrieval import DEFAULT_EMBEDDING_MODEL, build_index, load_records_for_index  # noqa: E402
from medical_rag_thesis.run_logging import run_with_logs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense retrieval index from normalized JSONL.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more normalized JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory for index files.")
    parser.add_argument("--backend", default="dense", choices=["dense", "tfidf"])
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=["topic", "question", "subquestion", "short_answer", "evidence"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--language", default="es", choices=["es", "eu"])
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    records = load_records_for_index(args.input)
    build_index(
        records,
        args.output_dir,
        model_name=args.model,
        backend=args.backend,
        text_fields=args.text_fields,
        batch_size=args.batch_size,
        language=args.language,
    )
    print(
        json.dumps(
            {
                "num_input_records": len(records),
                "output_dir": args.output_dir,
                "backend": args.backend,
                "model": args.model,
                "language": args.language,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parsed_args = parse_args()
    log_dir = Path(parsed_args.output_dir)
    run_with_logs(log_dir / "run.log", log_dir / "run.err", lambda: run(parsed_args))
