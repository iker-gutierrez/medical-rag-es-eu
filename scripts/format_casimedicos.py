#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import (  # noqa: E402
    canonical_name,
    clean_text,
    compact_record,
    ensure_id,
    find_column,
    read_table,
    records_to_dataframe,
    write_jsonl,
)


OPTION_COLUMN_PATTERNS = [
    re.compile(r"^(option|opcion|opción)_?([a-e])$"),
    re.compile(r"^([a-e])$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize CasiMedicos-style MCQ data.")
    parser.add_argument("--input", required=True, help="Raw CasiMedicos file.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--source", default="CasiMedicos")
    return parser.parse_args()


def option_columns(columns: list[str]) -> list[tuple[str, str]]:
    matched = []
    for column in columns:
        key = canonical_name(column)
        for pattern in OPTION_COLUMN_PATTERNS:
            match = pattern.match(key)
            if match:
                label = match.group(match.lastindex).upper()
                matched.append((label, column))
                break
    return sorted(matched)


def parse_options_cell(value: Any) -> dict[str, str]:
    text = clean_text(value)
    if not text:
        return {}
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return {str(key).upper(): clean_text(val) for key, val in parsed.items()}
        if isinstance(parsed, list):
            return {chr(65 + idx): clean_text(val) for idx, val in enumerate(parsed)}

    options: dict[str, str] = {}
    chunks = re.split(r"\n|;", text)
    for chunk in chunks:
        match = re.match(r"\s*([A-Ea-e])[\).\:-]\s*(.+)", chunk)
        if match:
            options[match.group(1).upper()] = clean_text(match.group(2))
    return options


def resolve_correct_answer(raw_answer: str, options: dict[str, str]) -> str:
    answer = clean_text(raw_answer)
    if len(answer) == 1 and answer.upper() in options:
        return options[answer.upper()]
    match = re.match(r"^([A-Ea-e])[\).\:-]?\s*(.*)$", answer)
    if match and match.group(1).upper() in options:
        suffix = clean_text(match.group(2))
        return suffix or options[match.group(1).upper()]
    return answer


def normalize_casimedicos(input_path: str, source: str) -> list[dict[str, Any]]:
    df = read_table(input_path)
    id_col = find_column(df, ["id", "sample_id", "question_id"], required=False)
    topic_col = find_column(df, ["topic", "tema", "subject", "asignatura"], required=False)
    question_col = find_column(df, ["Question", "question", "pregunta", "enunciado"])
    answer_col = find_column(
        df,
        ["correct answer", "correct_answer", "answer", "respuesta correcta", "respuesta"],
    )
    evidence_col = find_column(
        df,
        ["reasoning", "rationale", "explanation", "razonamiento", "explicacion", "explicación"],
        required=False,
    )
    options_col = find_column(df, ["options", "opciones"], required=False)
    option_cols = option_columns(list(df.columns))

    records = []
    for index, row in df.iterrows():
        options = parse_options_cell(row[options_col]) if options_col else {}
        for label, column in option_cols:
            value = clean_text(row[column])
            if value:
                options[label] = value
        raw_correct = clean_text(row[answer_col])
        short_answer = resolve_correct_answer(raw_correct, options)
        evidence = clean_text(row[evidence_col]) if evidence_col else ""
        record = compact_record(
            {
                "id": ensure_id("casimedicos", int(index), row[id_col] if id_col else None),
                "source": source,
                "topic": clean_text(row[topic_col]) if topic_col else "",
                "question": clean_text(row[question_col]),
                "options": options,
                "correct_answer": raw_correct,
                "short_answer": short_answer,
                "evidence": evidence,
            }
        )
        if record.get("question") and record.get("short_answer"):
            records.append(record)
    if not records:
        raise ValueError("No usable CasiMedicos rows found. Check column names and empty values.")
    return records


def main() -> None:
    args = parse_args()
    records = normalize_casimedicos(args.input, args.source)
    output = Path(args.output)
    write_jsonl(records, output)
    records_to_dataframe(records).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps({"num_records": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
