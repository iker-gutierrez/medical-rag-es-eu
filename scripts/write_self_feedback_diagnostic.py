#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("short_answer", "evidence", "overall")
SECTION_LABELS = {
    "short_answer": "answer",
    "evidence": "evidence",
    "overall": "overall",
}
METRICS = ("cosine_similarity", "bertscore_f1")
METRIC_LABELS = {
    "cosine_similarity": "Cosim",
    "bertscore_f1": "BERTScore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the v1 self-feedback diagnostic report.")
    parser.add_argument(
        "--metrics",
        default="reports/metrics/22_mistral7b_rag_no_think_e5_rerank3_v1_extractive_sf_dev.json",
    )
    parser.add_argument(
        "--predictions",
        default="experiments/runs/22_mistral7b_rag_no_think_e5_rerank3_v1_extractive_sf_dev/predictions.jsonl",
    )
    parser.add_argument("--output", default="reports/metrics/self_feedback_v1_diagnostic.md")
    return parser.parse_args()


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    text = f"{number:.2f}"
    if signed and number > 0:
        return "+" + text
    return text


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| section | self-feedback | Cosim | BERTScore |",
        "|---|---|---:|---:|",
    ]
    for section in SECTIONS:
        for condition, key in (
            ("noSF", "before_feedback"),
            ("SF", "after_feedback"),
            ("Delta", "self_feedback_delta"),
        ):
            metrics = summary[key][section]
            signed = condition == "Delta"
            lines.append(
                "| {section} | {condition} | {cosim} | {bert} |".format(
                    section=SECTION_LABELS[section],
                    condition=condition,
                    cosim=fmt(metrics.get("cosine_similarity"), signed=signed),
                    bert=fmt(metrics.get("bertscore_f1"), signed=signed),
                )
            )
    return lines


def old_v1_table() -> list[str]:
    rows = [
        ("Judgement", 85.3, 88.2, 2.9, 70.2, 72.7, 2.5),
        ("Evidence", 81.8, 95.0, 13.2, 62.5, 89.3, 26.8),
        ("Considerations", 85.2, 87.4, 2.2, 64.2, 65.7, 1.5),
        ("Overall", 84.1, 90.2, 6.1, 65.6, 75.9, 10.3),
    ]
    lines = [
        "| section | Cosim noSF | Cosim SF | Cosim Delta | BERTScore noSF | BERTScore SF | BERTScore Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for section, cosim_before, cosim_after, cosim_delta, bert_before, bert_after, bert_delta in rows:
        lines.append(
            f"| {section} | {cosim_before:.1f} | {cosim_after:.1f} | +{cosim_delta:.1f} | "
            f"{bert_before:.1f} | {bert_after:.1f} | +{bert_delta:.1f} |"
        )
    return lines


def contamination_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    def contains_marker(text: str) -> bool:
        lowered = text.lower()
        return "contexto:" in lowered or "\npregunta:" in lowered or "respuesta original:" in lowered

    initial = [record.get("initial_prediction_text") or "" for record in records]
    final = [record.get("prediction_text") or "" for record in records]
    parsed_final_evidence = [
        ((record.get("parsed_prediction") or {}).get("evidence") or "") for record in records
    ]
    return {
        "num_records": len(records),
        "initial_marker_count": sum(contains_marker(text) for text in initial),
        "final_marker_count": sum(contains_marker(text) for text in final),
        "final_evidence_marker_count": sum(contains_marker(text) for text in parsed_final_evidence),
    }


def write_report(metrics_path: Path, predictions_path: Path, output_path: Path) -> None:
    payload = load_json(metrics_path)
    summary = payload["summary"]
    records = read_jsonl(predictions_path)
    metadata_path = predictions_path.with_suffix(".meta.json")
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    stats = contamination_stats(records)
    before_overall_bert = summary["before_feedback"]["overall"].get("bertscore_f1")
    after_overall_bert = summary["after_feedback"]["overall"].get("bertscore_f1")
    evidence_delta_bert = summary["self_feedback_delta"]["evidence"].get("bertscore_f1")
    if stats["final_evidence_marker_count"]:
        formatting_note = (
            "- Prompt markers still appear inside parsed self-feedback evidence in "
            f"{stats['final_evidence_marker_count']} records, so parsing/formatting leakage may still hurt the score."
        )
    else:
        formatting_note = (
            "- The prompt-marker problem is fixed at parsing time: no parsed self-feedback evidence fields contain "
            "`Contexto`, `Pregunta`, or similar prompt markers."
        )

    lines = [
        "# Self-feedback v1 diagnostic",
        "",
        "This report checks self-feedback under the thesis evaluation pipeline.",
        "The thesis pipeline evaluates only the generated continuation, not the prompt plus continuation.",
        "",
        "## Compared runs",
        "",
        "| source | setting | examples | notes |",
        "|---|---|---:|---|",
        "| SNS_QA_RAG v1 reported table | notebook RAG + self-feedback | unclear from final table | The visible notebook code decodes the full sequence, so the evaluated prediction can include the prompt and retrieved context. |",
        f"| tested thesis run | `{metadata.get('experiment_name', '')}` | {summary.get('num_examples', '')} | Mistral-7B-Instruct; prompt stripped before scoring. |",
        "",
        "## Old v1 reported numbers",
        "",
        *old_v1_table(),
        "",
        "## Fixed continuation-only evaluation",
        "",
        "The table below uses the same 0-100 scale, but evaluates only generated text on the normalized thesis fields.",
        "",
        *fixed_table(summary),
        "",
        "## Leakage and formatting check",
        "",
        "| check | count |",
        "|---|---:|",
        f"| records | {stats['num_records']} |",
        f"| initial predictions containing prompt markers | {stats['initial_marker_count']} |",
        f"| self-feedback predictions containing prompt markers | {stats['final_marker_count']} |",
        f"| parsed self-feedback evidence containing prompt markers | {stats['final_evidence_marker_count']} |",
        "",
        "## Interpretation",
        "",
        "- The old v1 table shows large self-feedback gains, especially for evidence BERTScore.",
        "- In this thesis-pipeline run, self-feedback does not reproduce the old v1 gain; the continuation-only overall BERTScore changes from "
        f"{fmt(before_overall_bert)} to {fmt(after_overall_bert)}.",
        f"- Evidence BERTScore changes by {fmt(evidence_delta_bert, signed=True)}.",
        "- The most likely explanation is that the notebook evaluated decoded text that still contained the input prompt and retrieved context.",
        formatting_note,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_path = ROOT / args.metrics
    predictions_path = ROOT / args.predictions
    output_path = ROOT / args.output
    write_report(metrics_path, predictions_path, output_path)


if __name__ == "__main__":
    main()
