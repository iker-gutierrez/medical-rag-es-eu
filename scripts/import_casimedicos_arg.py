#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import records_to_dataframe, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and normalize the Spanish HiTZ/casimedicos-arg dataset."
    )
    parser.add_argument("--dataset", default="HiTZ/casimedicos-arg")
    parser.add_argument("--config", default="es")
    parser.add_argument("--raw-dir", default="data/raw/casimedicos_arg")
    parser.add_argument("--output-dir", default="data/processed/casimedicos")
    return parser.parse_args()


def parse_serialized_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unsupported serialized-list value: {type(value).__name__}")


def detokenize(tokens: Iterable[str]) -> str:
    text = " ".join(str(token) for token in tokens if token is not None)
    text = re.sub(r"\s+([,.;:?!%)\]])", r"\1", text)
    text = re.sub(r"([¿¡(\[])\s+", r"\1", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_label(labels: list[str]) -> str:
    clean = {label for label in labels if label}
    if any(label.endswith("Premise") for label in clean):
        return "Premise"
    if any(label.endswith("Claim") for label in clean):
        return "Claim"
    return "O"


def parsed_sentences(row: dict[str, Any]) -> list[dict[str, str]]:
    text = parse_serialized_list(row.get("text"))
    labels = parse_serialized_list(row.get("labels"))
    output = []
    for tokens, token_labels in zip(text, labels):
        if not tokens:
            continue
        sentence = detokenize(tokens)
        if not sentence:
            continue
        output.append(
            {
                "text": sentence,
                "label": sentence_label(token_labels or []),
            }
        )
    return output


def split_name(name: str) -> str:
    return "dev" if name == "validation" else name


def option_match(text: str) -> Optional[re.Match[str]]:
    return re.match(r"^\s*(\d+)\s*[-.)]\s*(.+)", text)


def normalize_record(row: dict[str, Any], split: str) -> Optional[dict[str, Any]]:
    sentences = parsed_sentences(row)
    if not sentences:
        return None

    topic = ""
    clinical_case: list[str] = []
    question = ""
    options: dict[str, str] = {}
    correct_answer = ""
    post_answer_evidence: list[str] = []
    seen_correct_answer = False

    for sentence in sentences:
        text = sentence["text"]
        label = sentence["label"]
        if text.upper().startswith("QUESTION TYPE:"):
            topic = text.split(":", 1)[1].strip()
            continue
        if text.upper().startswith("CLINICAL CASE"):
            continue
        correct_match = re.search(r"(?i)\bCORRECT ANSWER\s*:\s*([0-9A-Za-z]+)", text)
        if correct_match:
            correct_answer = correct_match.group(1).strip()
            seen_correct_answer = True
            continue
        match = option_match(text)
        if match and not seen_correct_answer:
            options[match.group(1)] = match.group(2).strip()
            continue
        if seen_correct_answer:
            if label in {"Claim", "Premise"}:
                post_answer_evidence.append(text)
            continue
        if "?" in text and not question:
            question = text
            continue
        if label == "Premise":
            clinical_case.append(text)

    if not question:
        question_candidates = [
            sentence["text"]
            for sentence in sentences
            if sentence["label"] == "O" and not sentence["text"].upper().startswith("QUESTION TYPE")
        ]
        question = question_candidates[-1] if question_candidates else ""

    question_parts = []
    if clinical_case:
        question_parts.append("Caso clinico: " + " ".join(clinical_case))
    if question:
        question_parts.append(question)
    normalized_question = " ".join(question_parts).strip()

    short_answer = options.get(correct_answer, correct_answer)
    evidence = " ".join(post_answer_evidence).strip()
    if not evidence:
        evidence = " ".join(clinical_case).strip()

    if not normalized_question or not short_answer:
        return None

    return {
        "id": f"casimedicos_{row.get('id')}",
        "source": "CasiMedicos",
        "split": split,
        "topic": topic,
        "question": normalized_question,
        "short_answer": short_answer,
        "evidence": evidence,
        "options": options,
        "correct_answer": correct_answer,
    }


def raw_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "split": split,
        "text": row.get("text"),
        "labels": row.get("labels"),
    }


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_dir / "all.jsonl")
    records_to_dataframe(records).to_csv(output_dir / "all.csv", index=False)
    for split in ("train", "dev", "test"):
        split_records = [record for record in records if record.get("split") == split]
        write_jsonl(split_records, output_dir / f"{split}.jsonl")
        records_to_dataframe(split_records).to_csv(output_dir / f"{split}.csv", index=False)


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    dataset = load_dataset(args.dataset, args.config)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    raw_records: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []

    for hf_split, split_dataset in dataset.items():
        split = split_name(hf_split)
        split_raw = []
        split_normalized = []
        for row in split_dataset:
            row_dict = dict(row)
            raw = raw_record(row_dict, split)
            split_raw.append(raw)
            normalized = normalize_record(row_dict, split)
            if normalized:
                split_normalized.append(normalized)
        raw_records.extend(split_raw)
        normalized_records.extend(split_normalized)
        write_jsonl(split_raw, raw_dir / f"{split}.jsonl")
        records_to_dataframe(split_raw).to_csv(raw_dir / f"{split}.csv", index=False)

    raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_records, raw_dir / "all.jsonl")
    records_to_dataframe(raw_records).to_csv(raw_dir / "all.csv", index=False)
    write_outputs(normalized_records, output_dir)

    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "num_raw_records": len(raw_records),
        "num_normalized_records": len(normalized_records),
        "splits": dict(Counter(record["split"] for record in normalized_records)),
        "sources": dict(Counter(record["source"] for record in normalized_records)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (raw_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
