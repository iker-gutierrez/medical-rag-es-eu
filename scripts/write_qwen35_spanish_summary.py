#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

METRICS_DIR = Path("reports/metrics")
OUT_MD = METRICS_DIR / "qwen35_4b_spanish_dev_results.md"
OUT_CSV = METRICS_DIR / "qwen35_4b_spanish_dev_results.csv"

RUNS = [
    {
        "dev_set": "SNS1064",
        "experiment": "Baseline LLM only",
        "reasoning": "no_think",
        "run_id": "94_qwen35_4b_no_rag_no_think_extractive_sns1064_dev",
    },
    {
        "dev_set": "SNS1064",
        "experiment": "Baseline LLM only",
        "reasoning": "think",
        "run_id": "95_qwen35_4b_no_rag_think_extractive_sns1064_dev",
    },
    {
        "dev_set": "SNS1064",
        "experiment": "3-shot + rerank top 5",
        "reasoning": "no_think",
        "run_id": "96_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_sns1064_dev",
    },
    {
        "dev_set": "SNS1064",
        "experiment": "3-shot + rerank top 5",
        "reasoning": "think",
        "run_id": "97_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_sns1064_dev",
    },
    {
        "dev_set": "CasiMedicos",
        "experiment": "Baseline LLM only",
        "reasoning": "no_think",
        "run_id": "98_qwen35_4b_no_rag_no_think_extractive_casimedicos_dev",
    },
    {
        "dev_set": "CasiMedicos",
        "experiment": "Baseline LLM only",
        "reasoning": "think",
        "run_id": "99_qwen35_4b_no_rag_think_extractive_casimedicos_dev",
    },
    {
        "dev_set": "CasiMedicos",
        "experiment": "e5 top 3",
        "reasoning": "no_think",
        "run_id": "100_qwen35_4b_rag_no_think_e5_topk3_extractive_casimedicos_dev",
    },
    {
        "dev_set": "CasiMedicos",
        "experiment": "e5 top 3",
        "reasoning": "think",
        "run_id": "101_qwen35_4b_rag_think_e5_topk3_extractive_casimedicos_dev",
    },
    {
        "dev_set": "SNS1064+CasiMedicos",
        "experiment": "Baseline LLM only",
        "reasoning": "no_think",
        "run_id": "102_qwen35_4b_no_rag_no_think_extractive_mixed_dev",
    },
    {
        "dev_set": "SNS1064+CasiMedicos",
        "experiment": "Baseline LLM only",
        "reasoning": "think",
        "run_id": "103_qwen35_4b_no_rag_think_extractive_mixed_dev",
    },
    {
        "dev_set": "SNS1064+CasiMedicos",
        "experiment": "3-shot + rerank top 5",
        "reasoning": "no_think",
        "run_id": "104_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_mixed_dev",
    },
    {
        "dev_set": "SNS1064+CasiMedicos",
        "experiment": "3-shot + rerank top 5",
        "reasoning": "think",
        "run_id": "105_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_mixed_dev",
    },
]


def nested_get(data: dict[str, Any], path: list[str]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    value = float(value)
    if signed and value >= 0:
        return f"+{value:.2f}"
    return f"{value:.2f}"


def load_summary(run_id: str) -> dict[str, Any] | None:
    path = METRICS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def row_for(spec: dict[str, str]) -> dict[str, Any]:
    summary = load_summary(spec["run_id"])
    row: dict[str, Any] = dict(spec)
    if summary is None:
        row.update(
            {
                "status": "missing",
                "answer_bertscore": None,
                "evidence_bertscore": None,
                "overall_bertscore": None,
                "answer_cosim": None,
                "evidence_cosim": None,
                "overall_cosim": None,
                "sec_sample": None,
                "input_tokens_sample": None,
                "output_tokens_sample": None,
                "tokens_sample": None,
            }
        )
        return row
    row.update(
        {
            "status": "done",
            "answer_bertscore": nested_get(summary, ["short_answer", "bertscore_f1"]),
            "evidence_bertscore": nested_get(summary, ["evidence", "bertscore_f1"]),
            "overall_bertscore": nested_get(summary, ["overall", "bertscore_f1"]),
            "answer_cosim": nested_get(summary, ["short_answer", "cosine_similarity"]),
            "evidence_cosim": nested_get(summary, ["evidence", "cosine_similarity"]),
            "overall_cosim": nested_get(summary, ["overall", "cosine_similarity"]),
            "sec_sample": nested_get(summary, ["cost", "timing", "example_seconds", "mean"]),
            "input_tokens_sample": nested_get(summary, ["cost", "token_counts", "input_tokens", "mean"]),
            "output_tokens_sample": nested_get(summary, ["cost", "token_counts", "output_tokens", "mean"]),
            "tokens_sample": nested_get(summary, ["cost", "token_counts", "total_tokens", "mean"]),
        }
    )
    return row


def add_deltas(rows: list[dict[str, Any]]) -> None:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_pair[(row["dev_set"], row["experiment"], row["reasoning"])] = row
    for row in rows:
        if row["reasoning"] != "think":
            row["bertscore_delta_vs_no_think"] = None
            row["cosim_delta_vs_no_think"] = None
            row["sec_delta_vs_no_think"] = None
            row["tokens_delta_vs_no_think"] = None
            continue
        base = by_pair.get((row["dev_set"], row["experiment"], "no_think"))
        if not base:
            continue
        row["bertscore_delta_vs_no_think"] = (
            None
            if row["overall_bertscore"] is None or base["overall_bertscore"] is None
            else row["overall_bertscore"] - base["overall_bertscore"]
        )
        row["cosim_delta_vs_no_think"] = (
            None
            if row["overall_cosim"] is None or base["overall_cosim"] is None
            else row["overall_cosim"] - base["overall_cosim"]
        )
        row["sec_delta_vs_no_think"] = (
            None
            if row["sec_sample"] is None or base["sec_sample"] is None
            else row["sec_sample"] - base["sec_sample"]
        )
        row["tokens_delta_vs_no_think"] = (
            None
            if row["tokens_sample"] is None or base["tokens_sample"] is None
            else row["tokens_sample"] - base["tokens_sample"]
        )


def markdown_table(rows: list[dict[str, Any]], dev_set: str) -> str:
    dev_rows = [row for row in rows if row["dev_set"] == dev_set]
    lines = [
        f"### {dev_set}",
        "",
        "| Experiment | Reasoning | BERT answer | BERT evidence | BERT overall | Δ BERT | Cosim overall | Δ Cosim | sec/sample | Δ sec | tokens/sample | Δ tokens | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in dev_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    row["reasoning"],
                    fmt(row["answer_bertscore"]),
                    fmt(row["evidence_bertscore"]),
                    fmt(row["overall_bertscore"]),
                    fmt(row.get("bertscore_delta_vs_no_think"), signed=True),
                    fmt(row["overall_cosim"]),
                    fmt(row.get("cosim_delta_vs_no_think"), signed=True),
                    fmt(row["sec_sample"]),
                    fmt(row.get("sec_delta_vs_no_think"), signed=True),
                    fmt(row["tokens_sample"]),
                    fmt(row.get("tokens_delta_vs_no_think"), signed=True),
                    row["status"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    rows = [row_for(spec) for spec in RUNS]
    add_deltas(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "dev_set",
        "experiment",
        "reasoning",
        "run_id",
        "status",
        "answer_bertscore",
        "evidence_bertscore",
        "overall_bertscore",
        "bertscore_delta_vs_no_think",
        "answer_cosim",
        "evidence_cosim",
        "overall_cosim",
        "cosim_delta_vs_no_think",
        "sec_sample",
        "sec_delta_vs_no_think",
        "input_tokens_sample",
        "output_tokens_sample",
        "tokens_sample",
        "tokens_delta_vs_no_think",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    sections = [
        "# Qwen3.5-4B Spanish dev ablation",
        "",
        "Dev-only comparison of `no_think` vs native Qwen `think` mode. Self-feedback is disabled in all runs.",
        "",
        markdown_table(rows, "SNS1064"),
        "",
        markdown_table(rows, "CasiMedicos"),
        "",
        markdown_table(rows, "SNS1064+CasiMedicos"),
        "",
    ]
    OUT_MD.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
