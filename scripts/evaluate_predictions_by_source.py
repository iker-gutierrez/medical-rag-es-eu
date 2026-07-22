#!/usr/bin/env python
"""Evaluate a mixed-corpus predictions.jsonl three ways: the full mixed set
(plain output), and each source subset (_sns1064, _casimedicos), by record id
prefix. Matches the file-naming convention every other mixed-dev run in this
thesis already uses (e.g. reports/metrics/1134_..._seed42_casimedicos.json),
which scripts/meanq.py and scripts/write_reasoning_latex_table.py read MC-acc
from -- MC-acc is defined only on CasiMedicos-Exp (multiple-choice), so a
mixed-table row with no _casimedicos file renders MC-acc and MeanQ as blank.

Usage:
    python scripts/evaluate_predictions_by_source.py \\
        --predictions experiments/runs/<run>/predictions.jsonl \\
        --output reports/metrics/<run>.json \\
        --bertscore-model bert-base-multilingual-cased --bertscore-lang es
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import read_jsonl, write_jsonl  # noqa: E402

SOURCE_PREFIXES = {"_sns1064": "sns1064", "_casimedicos": "casimedicos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True, help="Base output path; source suffixes are inserted before .json.")
    parser.add_argument("--semantic-model", default="")
    parser.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    parser.add_argument("--bertscore-lang", default="es")
    parser.add_argument(
        "--bertscore-device",
        default=None,
        help="Device for BERTScore, e.g. 'cuda:0'. Defaults to bert_score's own "
        "auto-detection when omitted.",
    )
    return parser.parse_args()


def run_eval(
    predictions_path: Path,
    output_path: Path,
    semantic_model: str,
    bertscore_model: str,
    bertscore_lang: str,
    bertscore_device: "str | None" = None,
) -> None:
    import subprocess

    command = [
        sys.executable, "scripts/evaluate_predictions.py",
        "--predictions", str(predictions_path),
        "--output", str(output_path),
        "--semantic-model", semantic_model,
        "--bertscore-model", bertscore_model,
        "--bertscore-lang", bertscore_lang,
    ]
    if bertscore_device:
        command.extend(["--bertscore-device", bertscore_device])
    subprocess.run(command, cwd=ROOT, check=True)


def _is_stale(output_path: Path, predictions_path: Path) -> bool:
    """True if output_path is missing or older than predictions_path. A
    metrics file that merely exists is not enough -- if it predates the
    predictions it is supposed to score (e.g. left over from a run that was
    later regenerated against a corrected corpus), it must be recomputed,
    not skipped."""
    if not output_path.exists():
        return True
    return output_path.stat().st_mtime < predictions_path.stat().st_mtime


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_path = Path(args.output)
    records = read_jsonl(predictions_path)

    if _is_stale(output_path, predictions_path):
        print(f"[eval-by-source] full mixed set -> {output_path}")
        run_eval(
            predictions_path, output_path, args.semantic_model, args.bertscore_model,
            args.bertscore_lang, args.bertscore_device,
        )
    else:
        print(f"[eval-by-source] SKIP (up to date): {output_path}")

    for suffix, prefix in SOURCE_PREFIXES.items():
        subset = [r for r in records if str(r.get("id", "")).startswith(prefix)]
        if not subset:
            print(f"[eval-by-source] SKIP {suffix} (no {prefix} records in {predictions_path})")
            continue
        subset_output = output_path.with_name(output_path.stem + suffix + ".json")
        if not _is_stale(subset_output, predictions_path):
            print(f"[eval-by-source] SKIP (up to date): {subset_output}")
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, dir=predictions_path.parent) as tmp:
            tmp_path = Path(tmp.name)
        write_jsonl(subset, tmp_path)
        # evaluate_predictions.py looks up predictions_path.with_suffix(".meta.json")
        # for the reference split; point the temp file's meta at the real one.
        real_meta = predictions_path.with_suffix(".meta.json")
        tmp_meta = tmp_path.with_suffix(".meta.json")
        if real_meta.exists():
            tmp_meta.write_text(real_meta.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[eval-by-source] {prefix} subset ({len(subset)} records) -> {subset_output}")
        try:
            run_eval(
                tmp_path, subset_output, args.semantic_model, args.bertscore_model,
                args.bertscore_lang, args.bertscore_device,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
            tmp_path.with_suffix(".log").unlink(missing_ok=True)
            tmp_path.with_suffix(".err").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
