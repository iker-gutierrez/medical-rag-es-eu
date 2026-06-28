#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
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
QUALITY_METRICS = (
    "token_precision",
    "token_recall",
    "token_overlap_f1",
    "rouge_l_f1",
    "cosine_similarity",
    "bertscore_f1",
    "answer_context_token_f1",
    "gold_context_token_f1",
)
QUALITY_METRIC_LABELS = {
    "token_precision": "Token_prec",
    "token_recall": "Token_rec",
    "token_overlap_f1": "token F1",
    "rouge_l_f1": "ROUGE-L",
    "cosine_similarity": "Cosim",
    "bertscore_f1": "BERTScore",
    "answer_context_token_f1": "answer-context F1",
    "gold_context_token_f1": "gold-context F1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the compact supervisor-facing dev summary.")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    parser.add_argument("--csv-output", default="reports/metrics/es_dev_ablation_results.csv")
    parser.add_argument("--markdown-output", default="reports/metrics/es_dev_ablation_results.md")
    return parser.parse_args()


def load_summary(metrics_dir: Path, filename: str) -> dict[str, Any] | None:
    path = metrics_dir / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("summary", {})


def no_sf_metrics(summary: dict[str, Any] | None, section: str = "overall") -> dict[str, Any]:
    if not summary:
        return {}
    return summary.get("before_feedback", {}).get(section) or summary.get(section, {})


def delta_metrics(summary: dict[str, Any] | None, section: str = "overall") -> dict[str, Any]:
    if not summary:
        return {}
    return summary.get("self_feedback_delta", {}).get(section, {})


def no_sf_cost(summary: dict[str, Any] | None) -> dict[str, float | None]:
    if not summary:
        return {
            "mean_example_seconds": None,
            "mean_input_tokens": None,
            "mean_output_tokens": None,
            "mean_total_tokens": None,
        }
    cost = summary.get("cost", {})
    timing = cost.get("timing", {})
    token_counts = cost.get("token_counts", {})

    def mean_timing(name: str) -> float:
        return float(timing.get(name, {}).get("mean") or 0.0)

    def mean_tokens(name: str) -> float:
        return float(token_counts.get(name, {}).get("mean") or 0.0)

    seconds = (
        mean_timing("retrieval_seconds")
        + mean_timing("rerank_seconds")
        + mean_timing("few_shot_seconds")
        + mean_timing("prompt_seconds")
        + mean_timing("generation_seconds")
    )
    input_tokens = mean_tokens("input_tokens")
    output_tokens = mean_tokens("initial_output_tokens")
    tokens = input_tokens + output_tokens
    return {
        "mean_example_seconds": seconds,
        "mean_input_tokens": input_tokens,
        "mean_output_tokens": output_tokens,
        "mean_total_tokens": tokens,
    }


def sf_cost(summary: dict[str, Any] | None) -> dict[str, float | None]:
    if not summary:
        return {
            "mean_example_seconds": None,
            "mean_input_tokens": None,
            "mean_output_tokens": None,
            "mean_total_tokens": None,
        }
    cost = summary.get("cost", {})
    timing = cost.get("timing", {})
    token_counts = cost.get("token_counts", {})

    def mean_timing(name: str) -> float:
        return float(timing.get(name, {}).get("mean") or 0.0)

    def mean_tokens(name: str) -> float:
        return float(token_counts.get(name, {}).get("mean") or 0.0)

    input_tokens = mean_tokens("input_tokens") + mean_tokens("feedback_input_tokens")
    output_tokens = mean_tokens("output_tokens")
    total_tokens = mean_tokens("total_tokens") or input_tokens + output_tokens
    return {
        "mean_example_seconds": mean_timing("example_seconds"),
        "mean_input_tokens": input_tokens,
        "mean_output_tokens": output_tokens,
        "mean_total_tokens": total_tokens,
    }


def delta_cost(summary: dict[str, Any] | None) -> dict[str, float | None]:
    if not summary:
        return {
            "mean_example_seconds": None,
            "mean_input_tokens": None,
            "mean_output_tokens": None,
            "mean_total_tokens": None,
        }
    before = no_sf_cost(summary)
    after = sf_cost(summary)
    return {
        key: None if before[key] is None or after[key] is None else after[key] - before[key]
        for key in before
    }


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    text = f"{number:.2f}"
    if signed and number > 0:
        return "+" + text
    return text


def cell_style(*, strong_left: bool = False) -> str:
    style = "border-left: 1px solid #d0d7de; border-right: 1px solid #d0d7de; padding: 4px 6px;"
    if strong_left:
        style += " border-left: 2px solid #57606a;"
    return style


def th(content: str, *, colspan: int | None = None, rowspan: int | None = None, strong_left: bool = False) -> str:
    attrs = [f'style="{cell_style(strong_left=strong_left)}"']
    if colspan is not None:
        attrs.append(f'colspan="{colspan}"')
    if rowspan is not None:
        attrs.append(f'rowspan="{rowspan}"')
    return f"<th {' '.join(attrs)}>{html.escape(content)}</th>"


def td(content: Any, *, strong_left: bool = False) -> str:
    return f'<td style="{cell_style(strong_left=strong_left)}">{html.escape(str(content))}</td>'


def make_row(
    *,
    row_id: str,
    experiment: str,
    status: str,
    metrics_file: str | None,
    completed_run: str,
    notes: str,
    metrics_dir: Path,
) -> dict[str, Any]:
    summary = load_summary(metrics_dir, metrics_file) if metrics_file else None
    cost = no_sf_cost(summary)
    if metrics_file and summary is None and status == "done":
        status = "submitted"
    row = {
        "id": row_id,
        "experiment": experiment,
        "status": status,
        "completed_run": completed_run,
        "mean_example_seconds": cost.get("mean_example_seconds"),
        "mean_input_tokens": cost.get("mean_input_tokens"),
        "mean_output_tokens": cost.get("mean_output_tokens"),
        "mean_total_tokens": cost.get("mean_total_tokens"),
        "notes": notes,
        "_summary": summary,
    }
    for section in SECTIONS:
        section_metrics = no_sf_metrics(summary, section)
        section_delta = delta_metrics(summary, section)
        for metric in QUALITY_METRICS:
            row[f"{section}_{metric}"] = section_metrics.get(metric)
        row[f"{section}_self_feedback_delta_bertscore"] = section_delta.get("bertscore_f1")
    return row


def feedback_summary_metrics(summary: dict[str, Any] | None, condition: str, section: str) -> dict[str, Any]:
    if not summary:
        return {}
    if condition == "noSF":
        return no_sf_metrics(summary, section)
    if condition == "SF":
        return summary.get("after_feedback", {}).get(section, {})
    if condition in {"Delta", "Δ"}:
        return delta_metrics(summary, section)
    return {}


def feedback_summary_cost(summary: dict[str, Any] | None, condition: str) -> dict[str, float | None]:
    if condition == "noSF":
        return no_sf_cost(summary)
    if condition == "SF":
        return sf_cost(summary)
    if condition in {"Delta", "Δ"}:
        return delta_cost(summary)
    return {
        "mean_example_seconds": None,
        "mean_input_tokens": None,
        "mean_output_tokens": None,
        "mean_total_tokens": None,
    }


def expand_feedback_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for row in rows:
        summary = row.get("_summary")
        conditions = ("noSF", "SF", "Δ") if summary else ("",)
        for condition in conditions:
            condition_row = dict(row)
            condition_row["self_feedback"] = condition
            cost = feedback_summary_cost(summary, condition)
            condition_row.update(cost)
            for section in SECTIONS:
                section_metrics = feedback_summary_metrics(summary, condition, section)
                for metric in QUALITY_METRICS:
                    condition_row[f"{section}_{metric}"] = section_metrics.get(metric)
            expanded.append(condition_row)
    return expanded


def best_model_id(rows: list[dict[str, Any]]) -> str | None:
    best_id, best_score = None, -1.0
    for row in rows:
        score = row.get("overall_bertscore_f1")
        if score is None:
            continue
        score = float(score)
        delta = row.get("overall_self_feedback_delta_bertscore")
        if delta is not None:
            score = max(score, score + float(delta))
        if score > best_score:
            best_id, best_score = str(row["id"]), score
    return best_id


def best_feedback_key(rows: list[dict[str, Any]]) -> tuple[str, str] | None:
    best_key, best_score = None, -1.0
    for row in rows:
        condition = row.get("self_feedback")
        if condition not in {"noSF", "SF"}:
            continue
        score = row.get("overall_bertscore_f1")
        if score is None:
            continue
        score = float(score)
        if score > best_score:
            best_key, best_score = (str(row["id"]), str(condition)), score
    return best_key


def build_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    return [
        make_row(
            row_id="0",
            experiment="Baseline LLM only",
            status="done",
            metrics_file="17_mistral7b_no_rag_no_think_extractive_sf_dev.json",
            completed_run="17_mistral7b_no_rag_no_think_extractive_sf_dev",
            notes="No retrieval, extractive-format output.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="1",
            experiment="Retriever top1, no reranker",
            status="done",
            metrics_file="32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev.json",
            completed_run="32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev",
            notes="e5 retrieves top 1 directly; no cross-encoder.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="2",
            experiment="Retriever top3, no reranker",
            status="done",
            metrics_file="30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev.json",
            completed_run="30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev",
            notes="e5 retrieves top 3 directly; no cross-encoder.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="3",
            experiment="Retriever top5, no reranker",
            status="done",
            metrics_file="33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev.json",
            completed_run="33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev",
            notes="e5 retrieves top 5 directly; no cross-encoder.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="4",
            experiment="Retriever top15 + reranker top1",
            status="done",
            metrics_file="28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev.json",
            completed_run="28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev",
            notes="e5 retrieves 15 candidates, cross-encoder keeps top 1.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="5",
            experiment="Retriever top15 + reranker top3",
            status="done",
            metrics_file="31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev.json",
            completed_run="31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev",
            notes="e5 retrieves 15 candidates, cross-encoder keeps top 3.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="6",
            experiment="Retriever top15 + reranker top5",
            status="done",
            metrics_file="29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev.json",
            completed_run="29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev",
            notes="e5 retrieves 15 candidates, cross-encoder keeps top 5.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="7",
            experiment="3-shot",
            status="done",
            metrics_file="20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev.json",
            completed_run="20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev",
            notes="Uses random SNS train examples to learn output format.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="8",
            experiment="Cross-domain retrieval",
            status="done",
            metrics_file="36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev.json",
            completed_run="36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev",
            notes="CasiMedicos-only index, e5 top15 + rerank top5, evaluated on SNS dev.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="9",
            experiment="Mixed-domain retrieval",
            status="done",
            metrics_file="37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev.json",
            completed_run="37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev",
            notes="SNS+CasiMedicos index, e5 top15 + rerank top5, evaluated on SNS dev.",
            metrics_dir=metrics_dir,
        ),
        make_row(
            row_id="10",
            experiment="3-shot + best retrieval configuration",
            status="done",
            metrics_file="38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev.json",
            completed_run="38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev",
            notes="3 random SNS train examples + SNS e5 top15 + rerank top5.",
            metrics_dir=metrics_dir,
        ),
    ]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "id",
        "experiment",
        "self_feedback",
        "status",
        "completed_run",
        "mean_example_seconds",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "notes",
    ]
    for section in SECTIONS:
        for metric in QUALITY_METRICS:
            fieldnames.append(f"{section}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(status_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]], path: Path) -> None:
    quality_span = len(QUALITY_METRICS) * len(SECTIONS)
    header_1 = (
        th("#", rowspan=3)
        + th("experiment", rowspan=3)
        + th("reasoning", rowspan=3)
        + th("Quality ↑", colspan=quality_span, strong_left=True)
        + th("Cost ↓", colspan=4, strong_left=True)
    )
    header_2 = "".join(
        th(QUALITY_METRIC_LABELS[metric], colspan=len(SECTIONS), strong_left=True)
        for metric in QUALITY_METRICS
    )
    header_2 += th("sec/sample", rowspan=2, strong_left=True)
    header_2 += th("tokens/sample", colspan=3, strong_left=True)
    header_3 = ""
    for _metric in QUALITY_METRICS:
        for index, section in enumerate(SECTIONS):
            header_3 += th(SECTION_LABELS[section], strong_left=index == 0)
    header_3 += th("input", strong_left=True) + th("output") + th("total")

    best_id = best_model_id(status_rows)
    best_key = best_feedback_key(result_rows)
    table_rows = []
    for row in result_rows:
        signed = row.get("self_feedback") in {"Delta", "Δ"}
        is_best = best_key is not None and (str(row["id"]), str(row["self_feedback"])) == best_key
        cells = [
            td(row["id"]),
            td(row["experiment"]),
            td(row["self_feedback"]),
        ]
        for metric in QUALITY_METRICS:
            for index, section in enumerate(SECTIONS):
                cells.append(td(fmt(row[f"{section}_{metric}"], signed=signed), strong_left=index == 0))
        cells.extend(
            [
                td(fmt(row["mean_example_seconds"], signed=signed), strong_left=True),
                td(fmt(row["mean_input_tokens"], signed=signed), strong_left=True),
                td(fmt(row["mean_output_tokens"], signed=signed)),
                td(fmt(row["mean_total_tokens"], signed=signed)),
            ]
        )
        row_style = ' style="background-color: #fff8c5; font-weight: 700;"' if is_best else ""
        table_rows.append(f"    <tr{row_style}>" + "".join(cells) + "</tr>")

    status_by_id = {str(row["id"]): row for row in status_rows}

    def summary_row(row_id: str) -> str:
        row = status_by_id[row_id]
        bert = fmt(row.get("overall_bertscore_f1"))
        delta = fmt(row.get("overall_self_feedback_delta_bertscore"), signed=True)
        if row_id == best_id:
            return f"| **{row_id}** | **{row['experiment']}** | **{bert}** | **{delta}** |"
        return f"| {row_id} | {row['experiment']} | {bert} | {delta} |"

    lines = [
        "# Dev ablation summary",
        "",
        "This is the compact supervisor-facing summary for the Spanish dev ablations. The archived previous table is in `dev_ablation_results_v1.md`.",
        "",
        "## Experimental setup",
        "",
        "- Dev sets: SNS1064, CasiMedicos, and SNS1064+CasiMedicos.",
        "- Generator: Mistral-7B-Instruct.",
        "- Main prompt family: cleaned extractive prompt.",
        "- Main retrieval stack: multilingual-e5 FAISS index + multilingual MS MARCO cross-encoder reranker.",
        "- Main reported row: no self-feedback. Self-feedback is tracked within each run.",
        "",
        "## Current results",
        "",
        "### SNS1064 dev results",
        "",
        "SNS1064 dev set, 106 examples, open-answer task.",
        "",
        "All quality scores are out of 100. Cost is experiment-level and is reported per sample.",
        "",
        '<table style="border-collapse: collapse; font-size: 0.9em;">',
        "  <thead>",
        f"    <tr>{header_1}</tr>",
        f"    <tr>{header_2}</tr>",
        f"    <tr>{header_3}</tr>",
        "  </thead>",
        "  <tbody>",
        *table_rows,
        "  </tbody>",
        "</table>",
    ]

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "### SNS1064 dev summary",
            "",
            "**Core findings (SNS1064 dev set, no self-feedback):**",
            "",
            "| # | Experiment | BERT | SF Δ |",
            "|---:|---|---:|---:|",
            summary_row("0"),
            "| | *Direct e5 retrieval (no reranker)* | | |",
            summary_row("1"),
            summary_row("2"),
            summary_row("3"),
            "| | *E5 top15 + cross-encoder reranker* | | |",
            summary_row("4"),
            summary_row("5"),
            summary_row("6"),
            "| | *Few-shot* | | |",
            summary_row("7"),
            summary_row("10"),
            "| | *Cross-domain retrieval (rerank top 5)* | | |",
            summary_row("8"),
            summary_row("9"),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_absolute():
        metrics_dir = ROOT / metrics_dir
    status_rows = build_rows(metrics_dir)
    result_rows = expand_feedback_rows(status_rows)
    write_csv(result_rows, ROOT / args.csv_output)
    write_markdown(status_rows, result_rows, ROOT / args.markdown_output)


if __name__ == "__main__":
    main()
