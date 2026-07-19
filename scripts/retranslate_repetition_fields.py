#!/usr/bin/env python
"""Retranslate a fixed, manually-confirmed list of (record_id, field) pairs
whose current Basque translation contains degenerate repetition (a MarianMT
failure mode distinct from the hard-truncation artefact of
sec:translation-artefact -- these fields are not below the truncation
ratio floor, but contain a 6+ word phrase repeated verbatim, e.g.
"kasuan, DM1aren kasuan, DM1aren kasuan, DM1aren" x28).

Scope is deliberately narrow and manual: sns1064/train (6 records) and
sns1064/test (3 records) only, per explicit instruction not to touch dev and
to only fix confirmed-severe cases, not every field with any repeated
n-gram (many short clinical phrases legitimately recur across a record,
e.g. dosage-comparison tables; those are not this bug).

Retranslates by forcing sentence-level packing (bypassing the
whole-paragraph fast path in translate_text, which is what let these long,
dense, number-heavy passages reach MarianMT as one large chunk in the first
place) and verifies the result no longer contains a repeated 6-gram before
accepting it, retrying at a smaller token budget if it does.

Usage:
  python scripts/retranslate_repetition_fields.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PROCESSED = ROOT / "data" / "processed"

from translate_to_basque import (  # noqa: E402
    load_model,
    pack_sentences,
    split_into_sentences,
    translate_batch,
)

# (es_name, eu_name, split, record_id, field) -- confirmed by direct
# inspection, not just the automatic n-gram heuristic (which also flags
# legitimate repeated clinical phrasing in tabular content).
TARGETS = [
    ("sns1064", "sns1064_eu", "train", "sns1064_00132", "evidence"),
    ("sns1064", "sns1064_eu", "train", "sns1064_00466", "evidence"),
    ("sns1064", "sns1064_eu", "train", "sns1064_00487", "evidence"),
    ("sns1064", "sns1064_eu", "train", "sns1064_00567", "evidence"),
    ("sns1064", "sns1064_eu", "train", "sns1064_00663", "evidence"),
    ("sns1064", "sns1064_eu", "train", "sns1064_00673", "evidence"),
    ("sns1064", "sns1064_eu", "test", "sns1064_00101", "evidence"),
    ("sns1064", "sns1064_eu", "test", "sns1064_00532", "evidence"),
    ("sns1064", "sns1064_eu", "test", "sns1064_00834", "evidence"),
]


def has_repetition(text: str, ngram: int = 10, min_count: int = 2) -> bool:
    """A long (10-word) exact-match repeat is degenerate looping; the source
    clinical text often legitimately reuses short templated phrasing (e.g.
    "Calidad de evidencia muy baja. Limitaciones debido a..." once per
    outcome, each with different values around it), which a shorter n-gram
    flags as a false positive."""
    words = text.split()
    if len(words) < ngram * min_count:
        return False
    ngrams = [" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
    counts = Counter(ngrams)
    return counts.most_common(1)[0][1] >= min_count


def translate_forced_sentence_packing(text: str, tokenizer, model, device: str, token_budget: int) -> str:
    """Translate by forcing sentence-level packing at the given token budget,
    regardless of whether the whole text already fits MarianMT's limit as one
    chunk. Re-implements translate_text's paragraph-split path without its
    whole-paragraph fast path, since that fast path is what let these long
    passages reach the model as a single oversized call."""
    import translate_to_basque as ttb

    original_budget = ttb.SENTENCE_TOKEN_BUDGET
    ttb.SENTENCE_TOKEN_BUDGET = token_budget
    try:
        paragraphs = [p for p in text.split("\n\n") if p.strip()] or [text]
        pieces: list[str] = []
        piece_is_paragraph_end: list[bool] = []
        for paragraph in paragraphs:
            groups = pack_sentences(split_into_sentences(paragraph), tokenizer)
            for j, group in enumerate(groups):
                pieces.append(group)
                piece_is_paragraph_end.append(j == len(groups) - 1)
        translated_pieces = translate_batch(pieces, tokenizer, model, device)
    finally:
        ttb.SENTENCE_TOKEN_BUDGET = original_budget

    out_parts: list[str] = []
    for i, (piece, is_end) in enumerate(zip(translated_pieces, piece_is_paragraph_end)):
        out_parts.append(piece)
        if i == len(translated_pieces) - 1:
            continue
        out_parts.append("\n\n" if is_end else " ")
    return "".join(out_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="HiTZ/medical_es-eu")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    by_file: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for es_name, eu_name, split, record_id, field in TARGETS:
        by_file.setdefault((es_name, eu_name), []).append((split, record_id, field))

    print(f"{len(TARGETS)} fields to retranslate (confirmed degenerate repetition)")
    for (es_name, eu_name), items in by_file.items():
        print(f"  {eu_name}: {len(items)} fields")

    if args.dry_run:
        return

    tokenizer, model = load_model(args.model, args.device)

    for (es_name, eu_name), items in by_file.items():
        by_split: dict[str, list[tuple[str, str]]] = {}
        for split, record_id, field in items:
            by_split.setdefault(split, []).append((record_id, field))

        for split, split_items in by_split.items():
            es_path = PROCESSED / es_name / f"{split}.jsonl"
            eu_path = PROCESSED / eu_name / f"{split}.jsonl"
            es_records = {json.loads(l)["id"]: json.loads(l) for l in es_path.open()}
            eu_lines = eu_path.read_text(encoding="utf-8").splitlines()
            eu_records = [json.loads(l) for l in eu_lines if l.strip()]
            eu_by_id = {r["id"]: r for r in eu_records}

            print(f"\n{eu_name}/{split}: retranslating {len(split_items)} fields")
            for i, (record_id, field) in enumerate(split_items, start=1):
                source_text = str(es_records[record_id].get(field) or "").strip()
                old = eu_by_id[record_id].get(field, "")

                new_translation = None
                for token_budget in (200, 120, 80):
                    candidate = translate_forced_sentence_packing(
                        source_text, tokenizer, model, args.device, token_budget
                    )
                    if not has_repetition(candidate):
                        new_translation = candidate
                        break
                    print(f"    still repetitive at token_budget={token_budget}, retrying finer")
                if new_translation is None:
                    print(f"  [{i}/{len(split_items)}] {record_id}.{field}: "
                          f"STILL REPETITIVE after all retries -- left unchanged, needs manual review")
                    continue

                eu_by_id[record_id][field] = new_translation
                print(
                    f"  [{i}/{len(split_items)}] {record_id}.{field}: "
                    f"{len(old)} -> {len(new_translation)} chars (source {len(source_text)} chars)"
                )

            eu_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in eu_records) + "\n",
                encoding="utf-8",
            )
            print(f"  saved {eu_path}")

    print("\nAll done. Rebuilding the combined sns1064_casimedicos_eu train file and "
          "the retrieval index is a separate step, needed only because a train record "
          "changed (see slurm/rebuild_eu_indices.sh); the 3 test-set fixes don't feed "
          "the retriever.")


if __name__ == "__main__":
    main()
