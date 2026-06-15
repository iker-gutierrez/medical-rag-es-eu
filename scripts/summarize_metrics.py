#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


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
SECTIONS = ("short_answer", "evidence", "overall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create summary tables from metrics JSON files.")
    parser.add_argument("--input", nargs="+", required=True, help="Metrics JSON files.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--markdown-output", help="Optional compact Markdown summary path.")
    return parser.parse_args()


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def experiment_name(path: Path) -> str:
    name = path.stem
    suffix = "_dev"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    sf_suffix = "_sf"
    if name.endswith(sf_suffix):
        name = name[: -len(sf_suffix)]
    topk3_aliases = {
        "05_mistral7b_rag_no_think": "05_mistral7b_rag_no_think_topk3",
        "07_mistral7b_rag_think": "07_mistral7b_rag_think_topk3",
    }
    name = topk3_aliases.get(name, name)
    return name


def add_section_metrics(
    row: dict[str, Any],
    summary: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    for section in SECTIONS:
        section_summary = summary.get(section, {})
        for metric in QUALITY_METRICS:
            row[f"{prefix}{section}_{metric}"] = section_summary.get(metric)


def cost_rows(cost: dict[str, Any]) -> dict[str, dict[str, Any]]:
    timing = cost.get("timing", {})
    token_counts = cost.get("token_counts", {})

    def mean_timing(name: str) -> float:
        value = nested_get(timing, name, "mean")
        return float(value or 0.0)

    def mean_tokens(name: str) -> float:
        value = nested_get(token_counts, name, "mean")
        return float(value or 0.0)

    no_sf_seconds = (
        mean_timing("retrieval_seconds")
        + mean_timing("rerank_seconds")
        + mean_timing("few_shot_seconds")
        + mean_timing("prompt_seconds")
        + mean_timing("generation_seconds")
    )
    sf_seconds = mean_timing("example_seconds")
    no_sf_input_tokens = mean_tokens("input_tokens")
    no_sf_output_tokens = mean_tokens("initial_output_tokens")
    no_sf_total_tokens = no_sf_input_tokens + no_sf_output_tokens
    sf_input_tokens = mean_tokens("input_tokens") + mean_tokens("feedback_input_tokens")
    sf_output_tokens = mean_tokens("output_tokens")
    sf_total_tokens = mean_tokens("total_tokens")
    return {
        "noSF": {
            "mean_example_seconds": no_sf_seconds,
            "mean_input_tokens": no_sf_input_tokens,
            "mean_output_tokens": no_sf_output_tokens,
            "mean_total_tokens": no_sf_total_tokens,
        },
        "SF": {
            "mean_example_seconds": sf_seconds,
            "mean_input_tokens": sf_input_tokens,
            "mean_output_tokens": sf_output_tokens,
            "mean_total_tokens": sf_total_tokens,
        },
        "Δ": {
            "mean_example_seconds": sf_seconds - no_sf_seconds,
            "mean_input_tokens": sf_input_tokens - no_sf_input_tokens,
            "mean_output_tokens": sf_output_tokens - no_sf_output_tokens,
            "mean_total_tokens": sf_total_tokens - no_sf_total_tokens,
        },
    }


def summarize_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    cost = summary.get("cost", {})
    base_row: dict[str, Any] = {
        "experiment": experiment_name(path),
        "metrics_file": str(path),
        "num_examples": summary.get("num_examples"),
    }

    if "before_feedback" in summary and "after_feedback" in summary:
        rows = []
        costs = cost_rows(cost)
        for condition, summary_key in (
            ("noSF", "before_feedback"),
            ("SF", "after_feedback"),
            ("Δ", "self_feedback_delta"),
        ):
            row = dict(base_row)
            row["self-feedback"] = condition
            add_section_metrics(row, summary.get(summary_key, {}))
            row.update(costs.get(condition, {}))
            rows.append(row)
        return rows

    row = dict(base_row)
    row["self-feedback"] = "noSF"
    add_section_metrics(row, summary)
    row.update(
        {
            "total_run_seconds": cost.get("total_run_seconds"),
            "model_load_seconds": cost.get("model_load_seconds"),
            "gpu_hours": cost.get("gpu_hours"),
            "mean_example_seconds": nested_get(cost, "timing", "example_seconds", "mean"),
            "median_example_seconds": nested_get(cost, "timing", "example_seconds", "median"),
            "mean_generation_seconds": nested_get(cost, "timing", "generation_seconds", "mean"),
            "mean_retrieval_seconds": nested_get(cost, "timing", "retrieval_seconds", "mean"),
            "mean_feedback_generation_seconds": nested_get(
                cost, "timing", "feedback_generation_seconds", "mean"
            ),
            "total_input_tokens": nested_get(cost, "token_counts", "input_tokens", "total"),
            "mean_input_tokens": nested_get(cost, "token_counts", "input_tokens", "mean"),
            "total_output_tokens": nested_get(cost, "token_counts", "output_tokens", "total"),
            "mean_output_tokens": nested_get(cost, "token_counts", "output_tokens", "mean"),
            "total_feedback_input_tokens": nested_get(
                cost, "token_counts", "feedback_input_tokens", "total"
            ),
            "mean_feedback_input_tokens": nested_get(
                cost, "token_counts", "feedback_input_tokens", "mean"
            ),
            "total_tokens": nested_get(cost, "token_counts", "total_tokens", "total"),
            "mean_total_tokens": nested_get(cost, "token_counts", "total_tokens", "mean"),
        }
    )
    return [row]


def format_number(value: Any, decimals: int = 2, *, signed: bool = False) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        if signed and value > 0:
            return f"+{value:,}"
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer() and abs(number) >= 1000:
        formatted = f"{int(number):,}"
    else:
        formatted = f"{number:.{decimals}f}"
    if signed and number > 0:
        return f"+{formatted}"
    return formatted


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    metric_groups = [
        ("token_precision", "token P"),
        ("token_recall", "token R"),
        ("token_overlap_f1", "token F1"),
        ("rouge_l_f1", "ROUGE-L"),
        ("cosine_similarity", "cosine"),
        ("bertscore_f1", "BERTScore"),
        ("answer_context_token_f1", "answer-context F1"),
        ("gold_context_token_f1", "gold-context F1"),
    ]
    section_labels = [
        ("short_answer", "answer"),
        ("evidence", "evidence"),
        ("overall", "overall"),
    ]
    cost_columns = [
        ("mean_example_seconds", "sec/sample"),
        ("mean_input_tokens", "input"),
        ("mean_output_tokens", "output"),
        ("mean_total_tokens", "total"),
    ]
    quality_colspan = len(metric_groups) * len(section_labels)
    table_style = "border-collapse: collapse; font-size: 0.9em;"
    cell_style = "border-left: 1px solid #d0d7de; border-right: 1px solid #d0d7de; padding: 4px 6px;"
    group_style = cell_style + " border-left: 2px solid #57606a;"

    def th(value: str, attrs: str = "", group_start: bool = False) -> str:
        style = group_style if group_start else cell_style
        return f"<th {attrs} style=\"{style}\">{html.escape(value)}</th>"

    def td(value: str, group_start: bool = False) -> str:
        style = group_style if group_start else cell_style
        return f"<td style=\"{style}\">{html.escape(value)}</td>"

    lines = ["# Dev ablation summary", ""]
    lines.append(f"<table style=\"{table_style}\">")
    lines.append("  <thead>")
    lines.append(
        "    <tr>"
        + th("experiment", 'rowspan="3"')
        + th("self-feedback", 'rowspan="3"')
        + th("Quality ↑", f'colspan="{quality_colspan}"', group_start=True)
        + th("Cost ↓", f'colspan="{len(cost_columns)}"', group_start=True)
        + "</tr>"
    )
    lines.append(
        "    <tr>"
        + "".join(th(label, 'colspan="3"', group_start=True) for _, label in metric_groups)
        + th("sec/sample", 'rowspan="2"', group_start=True)
        + th("tokens/sample", 'colspan="3"', group_start=True)
        + "</tr>"
    )
    lines.append(
        "    <tr>"
        + "".join(
            th(section, group_start=section_index == 0)
            for _ in metric_groups
            for section_index, (_, section) in enumerate(section_labels)
        )
        + "".join(
            th(label, group_start=key == "mean_input_tokens")
            for key, label in cost_columns
            if key != "mean_example_seconds"
        )
        + "</tr>"
    )
    lines.append("  </thead>")
    lines.append("  <tbody>")
    for row in rows:
        values = [str(row.get("experiment", "")), str(row.get("self-feedback", ""))]
        signed = row.get("self-feedback") == "Δ"
        for metric, _ in metric_groups:
            for section, _ in section_labels:
                values.append(
                    format_number(row.get(f"{section}_{metric}"), decimals=2, signed=signed)
                )
        for key, _ in cost_columns:
            decimals = 1 if key == "total_run_seconds" else 2
            values.append(format_number(row.get(key), decimals=decimals, signed=signed))
        cells = [td(values[0]), td(values[1])]
        quality_value_count = len(metric_groups) * len(section_labels)
        for index, value in enumerate(values[2:], start=1):
            if index <= quality_value_count:
                group_start = (index - 1) % len(section_labels) == 0
            else:
                group_start = index in {quality_value_count + 1, quality_value_count + 2}
            cells.append(td(value, group_start=group_start))
        lines.append("    <tr>" + "".join(cells) + "</tr>")
    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.input]
    rows = [row for path in paths for row in summarize_file(path)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    if args.markdown_output:
        markdown_output = Path(args.markdown_output)
        write_markdown(rows, markdown_output)
        print(f"Wrote {len(rows)} rows to {markdown_output}")


if __name__ == "__main__":
    main()
