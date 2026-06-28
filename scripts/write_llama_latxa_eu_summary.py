#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path("configs/experiments")
METRICS_DIR = Path("reports/metrics")
OUT_MD = METRICS_DIR / "llama_latxa_eu_dev_results.md"
OUT_CSV = METRICS_DIR / "llama_latxa_eu_dev_results.csv"

DEV_LABELS = {
    "sns1064_eu": "SNS1064 EU",
    "casimedicos_eu": "CasiMedicos EU",
    "mixed_eu": "SNS1064+CasiMedicos EU",
}

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "HiTZ/Latxa-Llama-3.1-8B-Instruct": "Latxa-Llama-3.1-8B-Instruct",
}
MODEL_ORDER = {
    "Llama-3.1-8B-Instruct": 0,
    "Latxa-Llama-3.1-8B-Instruct": 1,
}

EXPERIMENT_LABELS = {
    "no_rag": "Baseline LLM only",
    "rag_e5_topk1": "e5 top 1",
    "rag_e5_topk3": "e5 top 3",
    "rag_e5_topk5": "e5 top 5",
    "rag_e5_rerank1": "rerank top 1",
    "rag_e5_rerank3": "rerank top 3",
    "rag_e5_rerank5": "rerank top 5",
    "3shot_no_rag": "3-shot, no RAG",
    "rag_3shot_e5_rerank5": "3-shot + rerank top 5",
    "rag_cross_domain_e5_rerank5": "cross-domain retrieval",
    "rag_mixed_e5_rerank5": "mixed-domain retrieval",
    "rag_sns1064_e5_rerank5": "SNS-only retrieval",
    "rag_casimedicos_e5_rerank5": "CasiMedicos-only retrieval",
}

ORDER = [
    "no_rag",
    "rag_e5_topk1",
    "rag_e5_topk3",
    "rag_e5_topk5",
    "rag_e5_rerank1",
    "rag_e5_rerank3",
    "rag_e5_rerank5",
    "3shot_no_rag",
    "rag_3shot_e5_rerank5",
    "rag_cross_domain_e5_rerank5",
    "rag_mixed_e5_rerank5",
    "rag_sns1064_e5_rerank5",
    "rag_casimedicos_e5_rerank5",
]


def infer_dataset_slug(experiment_name: str) -> str:
    for slug in sorted(DEV_LABELS, key=len, reverse=True):
        if experiment_name.endswith(f"_{slug}"):
            return slug
    return ""


def infer_model_slug(experiment_name: str) -> str:
    if experiment_name.startswith("latxa_llama31_8b_"):
        return "latxa_llama31_8b"
    if experiment_name.startswith("llama31_8b_"):
        return "llama31_8b"
    return ""


def infer_experiment_slug(experiment_name: str) -> str:
    dataset_slug = infer_dataset_slug(experiment_name)
    model_slug = infer_model_slug(experiment_name)
    prefix = f"{model_slug}_"
    suffix = f"_extractive_{dataset_slug}"
    if not model_slug or not dataset_slug or not experiment_name.startswith(prefix):
        return experiment_name
    return experiment_name[len(prefix) : -len(suffix)]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    experiment_name = config["experiment_name"]
    return {
        "run_id": path.stem,
        "model": config["model"],
        "model_label": MODEL_LABELS.get(config["model"], config["model"]),
        "dataset_slug": infer_dataset_slug(experiment_name),
        "dev_set": DEV_LABELS.get(infer_dataset_slug(experiment_name), infer_dataset_slug(experiment_name)),
        "experiment_slug": infer_experiment_slug(experiment_name),
        "experiment": EXPERIMENT_LABELS.get(infer_experiment_slug(experiment_name), infer_experiment_slug(experiment_name)),
    }


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


def metric_row(spec: dict[str, Any], summary: dict[str, Any] | None, condition: str) -> dict[str, Any]:
    row = dict(spec)
    row["self_feedback"] = condition
    if summary is None:
        row.update(empty_metrics("missing"))
        return row

    if condition == "noSF":
        metrics = summary.get("before_feedback", summary)
        cost = summary.get("cost", {})
        signed = False
    elif condition == "SF":
        metrics = summary.get("after_feedback", summary)
        cost = summary.get("cost", {})
        signed = False
    else:
        metrics = summary.get("self_feedback_delta")
        cost = delta_cost(summary)
        signed = True
        if not metrics:
            row.update(empty_metrics("missing"))
            return row

    row.update(
        {
            "status": "done",
            "signed": signed,
            "answer_bertscore": nested_get(metrics, ["short_answer", "bertscore_f1"]),
            "evidence_bertscore": nested_get(metrics, ["evidence", "bertscore_f1"]),
            "overall_bertscore": nested_get(metrics, ["overall", "bertscore_f1"]),
            "answer_cosim": nested_get(metrics, ["short_answer", "cosine_similarity"]),
            "evidence_cosim": nested_get(metrics, ["evidence", "cosine_similarity"]),
            "overall_cosim": nested_get(metrics, ["overall", "cosine_similarity"]),
            "sec_sample": nested_get(cost, ["timing", "example_seconds", "mean"]),
            "input_tokens_sample": nested_get(cost, ["token_counts", "input_tokens", "mean"]),
            "output_tokens_sample": nested_get(cost, ["token_counts", "output_tokens", "mean"]),
            "tokens_sample": nested_get(cost, ["token_counts", "total_tokens", "mean"]),
        }
    )
    return row


def empty_metrics(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "signed": False,
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


def delta_cost(summary: dict[str, Any]) -> dict[str, Any]:
    cost = summary.get("cost", {})
    timing = cost.get("timing", {})
    tokens = cost.get("token_counts", {})
    return {
        "timing": {
            "example_seconds": {
                "mean": nested_get(timing, ["feedback_generation_seconds", "mean"]) or 0.0,
            }
        },
        "token_counts": {
            "input_tokens": {
                "mean": nested_get(tokens, ["feedback_input_tokens", "mean"]) or 0.0,
            },
            "output_tokens": {
                "mean": (nested_get(tokens, ["output_tokens", "mean"]) or 0.0)
                - (nested_get(tokens, ["initial_output_tokens", "mean"]) or 0.0),
            },
            "total_tokens": {
                "mean": (nested_get(tokens, ["feedback_input_tokens", "mean"]) or 0.0)
                + (nested_get(tokens, ["output_tokens", "mean"]) or 0.0)
                - (nested_get(tokens, ["initial_output_tokens", "mean"]) or 0.0),
            },
        },
    }


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(CONFIG_DIR.glob("1*_*.json")):
        run_number = int(config_path.name.split("_", 1)[0])
        if run_number < 106 or run_number > 171:
            continue
        spec = load_config(config_path)
        summary = load_summary(spec["run_id"])
        rows.append(metric_row(spec, summary, "noSF"))
        rows.append(metric_row(spec, summary, "SF"))
        rows.append(metric_row(spec, summary, "Δ"))
    return sorted(
        rows,
        key=lambda row: (
            list(DEV_LABELS.values()).index(row["dev_set"]),
            MODEL_ORDER.get(row["model_label"], 999),
            ORDER.index(row["experiment_slug"]) if row["experiment_slug"] in ORDER else 999,
            {"noSF": 0, "SF": 1, "Δ": 2}[row["self_feedback"]],
        ),
    )


def markdown_table(rows: list[dict[str, Any]], dev_set: str) -> str:
    dev_rows = [row for row in rows if row["dev_set"] == dev_set]
    lines = [
        f"### {dev_set}",
        "",
        "| Experiment | Model | self-feedback | BERT answer | BERT evidence | BERT overall | Cosim overall | sec/sample | input tok/sample | output tok/sample | total tok/sample | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in dev_rows:
        signed = bool(row.get("signed"))
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    row["model_label"],
                    row["self_feedback"],
                    fmt(row["answer_bertscore"], signed=signed),
                    fmt(row["evidence_bertscore"], signed=signed),
                    fmt(row["overall_bertscore"], signed=signed),
                    fmt(row["overall_cosim"], signed=signed),
                    fmt(row["sec_sample"], signed=signed),
                    fmt(row["input_tokens_sample"], signed=signed),
                    fmt(row["output_tokens_sample"], signed=signed),
                    fmt(row["tokens_sample"], signed=signed),
                    row["status"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    rows = build_rows()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "dev_set",
        "experiment",
        "model_label",
        "self_feedback",
        "run_id",
        "status",
        "answer_bertscore",
        "evidence_bertscore",
        "overall_bertscore",
        "answer_cosim",
        "evidence_cosim",
        "overall_cosim",
        "sec_sample",
        "input_tokens_sample",
        "output_tokens_sample",
        "tokens_sample",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    sections = [
        "# Llama vs Latxa-Llama Basque dev ablation",
        "",
        "Dev-only comparison of `meta-llama/Llama-3.1-8B-Instruct` and `HiTZ/Latxa-Llama-3.1-8B-Instruct` across the full Basque ablation grid. Because these models do not expose a native thinking mode like Qwen, self-feedback is enabled for every run.",
        "",
        markdown_table(rows, "SNS1064 EU"),
        "",
        markdown_table(rows, "CasiMedicos EU"),
        "",
        markdown_table(rows, "SNS1064+CasiMedicos EU"),
        "",
    ]
    OUT_MD.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
