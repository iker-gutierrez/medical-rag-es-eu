#!/usr/bin/env python3
"""
Translate Spanish medical JSONL datasets to Basque using HiTZ/medical_es-eu.

Translates text fields in-place; keeps IDs, splits, and numeric keys unchanged.
Long texts are split by double-newline to stay within the model's 512-token limit.

Usage:
  python scripts/translate_to_basque.py \
      --input  data/processed/sns1064/train.jsonl \
      --output data/processed/sns1064_eu/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TRANSLATABLE_FIELDS = ("topic", "question", "subquestion", "short_answer", "evidence")
MAX_CHARS = 1800  # rough guard; MarianMT max is ~512 tokens ≈ 2000 chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="Input JSONL file(s).")
    parser.add_argument("--output", required=True, nargs="+", help="Output JSONL file(s).")
    parser.add_argument("--model", default="HiTZ/medical_es-eu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(model_name: str, device: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading translation model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer: Any, model: Any, device: str) -> list[str]:
    import torch

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


def translate_text(text: str, tokenizer: Any, model: Any, device: str) -> str:
    """Translate a single text, chunking by paragraph if needed."""
    if not text or not text.strip():
        return text
    if len(text) <= MAX_CHARS:
        return translate_batch([text], tokenizer, model, device)[0]
    # chunk by double-newline (paragraph break)
    chunks = [c for c in text.split("\n\n") if c.strip()]
    if not chunks:
        return text
    translated_chunks = translate_batch(chunks, tokenizer, model, device)
    return "\n\n".join(translated_chunks)


def collect_texts(records: list[dict[str, Any]]) -> tuple[list[str], list[tuple[int, str]]]:
    """Return a flat list of texts and their (record_idx, field) locations."""
    texts: list[str] = []
    locations: list[tuple[int, str]] = []
    for i, record in enumerate(records):
        for field in TRANSLATABLE_FIELDS:
            value = record.get(field)
            if value and isinstance(value, str) and value.strip():
                texts.append(value.strip())
                locations.append((i, field))
        # options: translate string values, skip "nan" and numeric-only strings
        options = record.get("options")
        if isinstance(options, dict):
            for key, val in options.items():
                if isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
                    texts.append(val.strip())
                    locations.append((i, f"options.{key}"))
    return texts, locations


def run_translation(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> list[str]:
    results: list[str] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        # split long texts individually
        translated: list[str] = []
        short, short_idx = [], []
        for j, t in enumerate(batch):
            if len(t) <= MAX_CHARS:
                short.append(t)
                short_idx.append(j)
            else:
                translated.append((j, translate_text(t, tokenizer, model, device)))
        if short:
            batch_out = translate_batch(short, tokenizer, model, device)
            for j, out in zip(short_idx, batch_out):
                translated.append((j, out))
        translated.sort()
        results.extend(out for _, out in translated)
        print(f"  Translated {min(start + batch_size, total)}/{total}", flush=True)
    return results


def apply_translations(
    records: list[dict[str, Any]],
    locations: list[tuple[int, str]],
    translations: list[str],
) -> None:
    for (rec_idx, field), translation in zip(locations, translations):
        if field.startswith("options."):
            key = field[len("options."):]
            records[rec_idx]["options"][key] = translation
        else:
            records[rec_idx][field] = translation


def process_file(
    input_path: Path,
    output_path: Path,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> None:
    print(f"\n{input_path} → {output_path}", flush=True)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts, locations = collect_texts(records)
    print(f"  {len(records)} records, {len(texts)} text fields to translate", flush=True)
    translations = run_translation(texts, tokenizer, model, device, batch_size)
    apply_translations(records, locations, translations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved {len(records)} records to {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.output):
        print("ERROR: --input and --output must have the same number of paths.", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = load_model(args.model, args.device)

    for inp, out in zip(args.input, args.output):
        process_file(Path(inp), Path(out), tokenizer, model, args.device, args.batch_size)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
