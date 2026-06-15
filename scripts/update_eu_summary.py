"""
Appends Basque (EU) result sections to dev_ablation_results.md,
using the same HTML table format as the Spanish sections.
Run after eval_eu.sh completes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from write_supervisor_summary import (  # noqa: E402
    QUALITY_METRICS,
    QUALITY_METRIC_LABELS,
    SECTION_LABELS,
    SECTIONS,
    expand_feedback_rows,
    fmt as fmt_full,
    make_row,
    td,
    th,
)

METRICS_DIR = Path("reports/metrics")
SUMMARY_PATH = Path("reports/metrics/dev_ablation_results.md")

# ── experiment definitions ────────────────────────────────────────────────────

SNS_EU_EXPS = {
    0:  "61_latxa7b_no_rag_extractive_sf_sns1064_eu_dev",
    1:  "62_latxa7b_rag_e5_topk1_extractive_sf_sns1064_eu_dev",
    2:  "63_latxa7b_rag_e5_topk3_extractive_sf_sns1064_eu_dev",
    3:  "64_latxa7b_rag_e5_topk5_extractive_sf_sns1064_eu_dev",
    4:  "65_latxa7b_rag_e5_rerank1_extractive_sf_sns1064_eu_dev",
    5:  "66_latxa7b_rag_e5_rerank3_extractive_sf_sns1064_eu_dev",
    6:  "67_latxa7b_rag_e5_rerank5_extractive_sf_sns1064_eu_dev",
    7:  "68_latxa7b_3shot_no_rag_extractive_sf_sns1064_eu_dev",
    8:  "69_latxa7b_rag_3shot_e5_rerank5_extractive_sf_sns1064_eu_dev",
    9:  "70_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_sns1064_eu_dev",
    10: "71_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_sns1064_eu_dev",
}

CASI_EU_EXPS = {
    0:  "72_latxa7b_no_rag_extractive_sf_casimedicos_eu_dev",
    1:  "73_latxa7b_rag_e5_topk1_extractive_sf_casimedicos_eu_dev",
    2:  "74_latxa7b_rag_e5_topk3_extractive_sf_casimedicos_eu_dev",
    3:  "75_latxa7b_rag_e5_topk5_extractive_sf_casimedicos_eu_dev",
    4:  "76_latxa7b_rag_e5_rerank1_extractive_sf_casimedicos_eu_dev",
    5:  "77_latxa7b_rag_e5_rerank3_extractive_sf_casimedicos_eu_dev",
    6:  "78_latxa7b_rag_e5_rerank5_extractive_sf_casimedicos_eu_dev",
    7:  "79_latxa7b_3shot_no_rag_extractive_sf_casimedicos_eu_dev",
    8:  "80_latxa7b_rag_3shot_e5_rerank5_extractive_sf_casimedicos_eu_dev",
    9:  "81_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev",
    10: "82_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev",
}

MIXED_EU_EXPS = {
    0:  "83_latxa7b_no_rag_extractive_sf_mixed_eu_dev",
    1:  "84_latxa7b_rag_e5_topk1_extractive_sf_mixed_eu_dev",
    2:  "85_latxa7b_rag_e5_topk3_extractive_sf_mixed_eu_dev",
    3:  "86_latxa7b_rag_e5_topk5_extractive_sf_mixed_eu_dev",
    4:  "87_latxa7b_rag_e5_rerank1_extractive_sf_mixed_eu_dev",
    5:  "88_latxa7b_rag_e5_rerank3_extractive_sf_mixed_eu_dev",
    6:  "89_latxa7b_rag_e5_rerank5_extractive_sf_mixed_eu_dev",
    7:  "90_latxa7b_3shot_no_rag_extractive_sf_mixed_eu_dev",
    8:  "91_latxa7b_rag_3shot_e5_rerank5_extractive_sf_mixed_eu_dev",
    9:  "92_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_mixed_eu_dev",
    10: "93_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_mixed_eu_dev",
}

SNS_EU_LABELS = {
    0: "Baseline LLM only",
    1: "e5 top 1 (SNS EU index)",
    2: "e5 top 3 (SNS EU index)",
    3: "e5 top 5 (SNS EU index)",
    4: "rerank top 1 (SNS EU index)",
    5: "rerank top 3 (SNS EU index)",
    6: "rerank top 5 (SNS EU index)",
    7: "3-shot, no RAG",
    8: "3-shot + rerank top 5 (SNS EU index)",
    9: "Cross-domain: CasiMedicos EU index",
    10: "Mixed-domain: SNS+CasiMedicos EU index",
}

CASI_EU_LABELS = {
    0: "Baseline LLM only",
    1: "e5 top 1 (CasiMedicos EU index)",
    2: "e5 top 3 (CasiMedicos EU index)",
    3: "e5 top 5 (CasiMedicos EU index)",
    4: "rerank top 1 (CasiMedicos EU index)",
    5: "rerank top 3 (CasiMedicos EU index)",
    6: "rerank top 5 (CasiMedicos EU index)",
    7: "3-shot, no RAG",
    8: "3-shot + rerank top 5 (CasiMedicos EU index)",
    9: "Cross-domain: SNS EU index",
    10: "Mixed-domain: SNS+CasiMedicos EU index",
}

MIXED_EU_LABELS = {
    0: "Baseline LLM only",
    1: "e5 top 1 (mixed EU index)",
    2: "e5 top 3 (mixed EU index)",
    3: "e5 top 5 (mixed EU index)",
    4: "rerank top 1 (mixed EU index)",
    5: "rerank top 3 (mixed EU index)",
    6: "rerank top 5 (mixed EU index)",
    7: "3-shot, no RAG",
    8: "3-shot + rerank top 5 (mixed EU index)",
    9: "Cross-domain: SNS EU index",
    10: "Cross-domain: CasiMedicos EU index",
}


def fmt(v: float) -> str:
    return f"{v:.2f}"


def sign(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


def parenthetical_label(label: str) -> str:
    if label.endswith(")") and " (" in label:
        base, detail = label.rsplit(" (", 1)
        return f"{base}, {detail[:-1]}"
    return label


def best_rag_exp(exps: dict) -> int:
    best_idx, best_val = -1, -1.0
    for idx in range(1, 7):
        path = METRICS_DIR / f"{exps[idx]}.json"
        if not path.exists():
            continue
        val = json.loads(path.read_text())["summary"]["before_feedback"]["overall"]["bertscore_f1"]
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx


def best_model_exp(exps: dict) -> int:
    best_idx, best_val = -1, -1.0
    for idx, run in exps.items():
        path = METRICS_DIR / f"{run}.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())["summary"]
        val = summary["before_feedback"]["overall"]["bertscore_f1"]
        delta = summary.get("self_feedback_delta", {}).get("overall", {}).get("bertscore_f1")
        if delta is not None:
            val = max(val, val + delta)
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx


def summary_row(idx: int, label: str, bert: str, delta: str, best_idx: int) -> str:
    if idx == best_idx:
        return f"| **{idx}** | **{label}** | **{bert}** | **{delta}** |"
    return f"| {idx} | {label} | {bert} | {delta} |"


def best_feedback_key(rows: list[dict]) -> tuple[str, str] | None:
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


def build_metric_table(result_rows: list[dict]) -> str:
    quality_span = len(QUALITY_METRICS) * len(SECTIONS)
    header_1 = (
        th("#", rowspan=3)
        + th("experiment", rowspan=3)
        + th("self-feedback", rowspan=3)
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
                cells.append(td(fmt_full(row[f"{section}_{metric}"], signed=signed), strong_left=index == 0))
        cells.extend([
            td(fmt_full(row["mean_example_seconds"], signed=signed), strong_left=True),
            td(fmt_full(row["mean_input_tokens"], signed=signed), strong_left=True),
            td(fmt_full(row["mean_output_tokens"], signed=signed)),
            td(fmt_full(row["mean_total_tokens"], signed=signed)),
        ])
        row_style = ' style="background-color: #fff8c5; font-weight: 700;"' if is_best else ""
        table_rows.append(f"    <tr{row_style}>" + "".join(cells) + "</tr>")

    return "\n".join([
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
    ])


def build_full_section(title: str, dev_desc: str, exps: dict, labels: dict) -> str:
    best_idx = best_rag_exp(exps)
    best_note = (
        f"Best RAG config (exps 1–6): **exp {best_idx}** ({parenthetical_label(labels[best_idx])})."
        if best_idx >= 0 else "Best RAG config pending."
    )

    status_rows = [
        make_row(
            row_id=str(idx),
            experiment=labels[idx],
            status="done" if (METRICS_DIR / f"{exps[idx]}.json").exists() else "missing",
            metrics_file=f"{exps[idx]}.json",
            completed_run=exps[idx],
            notes="",
            metrics_dir=METRICS_DIR,
        )
        for idx in range(11)
    ]
    result_rows = expand_feedback_rows(status_rows)
    table = build_metric_table(result_rows)

    return f"""
### {title}

*{dev_desc}. Model: Latxa-7B. {best_note}*

All quality scores are out of 100. Cost is experiment-level and is reported per sample.

{table}
"""


def build_summary_section(title: str, dev_desc: str, exps: dict, labels: dict) -> str:
    rows = []
    best_idx = best_model_exp(exps)
    for idx in range(11):
        path = METRICS_DIR / f"{exps[idx]}.json"
        if not path.exists():
            rows.append(f"| {idx} | {labels[idx]} | missing | missing |")
            continue
        d = json.loads(path.read_text())["summary"]
        bert = d["before_feedback"]["overall"]["bertscore_f1"]
        delta = d["self_feedback_delta"]["overall"]["bertscore_f1"]
        rows.append(summary_row(idx, labels[idx], fmt(bert), sign(delta), best_idx))

    best_idx = best_rag_exp(exps)
    best_label = f"exp {best_idx} ({parenthetical_label(labels[best_idx])})" if best_idx >= 0 else "exp 6 (assumed)"

    grouped = [
        rows[0],
        "| | *Direct e5 retrieval (no reranker)* | | |",
        *rows[1:4],
        "| | *E5 top15 + cross-encoder reranker* | | |",
        *rows[4:7],
        "| | *Few-shot* | | |",
        *rows[7:9],
        "| | *Cross-domain / mixed-domain retrieval* | | |",
        *rows[9:11],
    ]

    return f"""
### {title}

*Evaluation set: {dev_desc}. Model: Latxa-7B. Best RAG config from exps 1–6: {best_label}.*

| # | Experiment | BERT | SF Δ |
|---:|---|---:|---:|
{chr(10).join(grouped)}
"""


EU_FULL_MARKER = "\n## Basque (EU) results\n"
EU_SUMMARY_MARKER = "\n### EU summary\n"


def main() -> None:
    content = SUMMARY_PATH.read_text(encoding="utf-8")

    # idempotent: remove previous EU sections before re-appending
    if EU_FULL_MARKER in content:
        content = content[: content.index(EU_FULL_MARKER)]

    sns_eu_full    = build_full_section(
        "SNS1064 EU dev results",
        "SNS1064 EU dev set, 106 examples, open-answer task",
        SNS_EU_EXPS, SNS_EU_LABELS,
    )
    casi_eu_full   = build_full_section(
        "CasiMedicos EU dev results",
        "CasiMedicos EU dev set, 55 examples, multiple-choice task",
        CASI_EU_EXPS, CASI_EU_LABELS,
    )
    mixed_eu_full  = build_full_section(
        "SNS1064+CasiMedicos EU dev results",
        "SNS1064+CasiMedicos EU dev set, 161 examples, mixed open-answer + multiple-choice",
        MIXED_EU_EXPS, MIXED_EU_LABELS,
    )

    sns_eu_summary   = build_summary_section(
        "SNS1064 EU dev summary",
        "SNS1064 EU dev set, 106 examples",
        SNS_EU_EXPS, SNS_EU_LABELS,
    )
    casi_eu_summary  = build_summary_section(
        "CasiMedicos EU dev summary",
        "CasiMedicos EU dev set, 55 examples",
        CASI_EU_EXPS, CASI_EU_LABELS,
    )
    mixed_eu_summary = build_summary_section(
        "SNS1064+CasiMedicos EU dev summary",
        "SNS1064+CasiMedicos EU dev set, 161 examples",
        MIXED_EU_EXPS, MIXED_EU_LABELS,
    )

    # insert full HTML tables before the Takeaways section,
    # and compact markdown summaries inside Takeaways, using the same pattern as Spanish
    takeaways_marker = "\n## Takeaways\n"
    if takeaways_marker not in content:
        raise RuntimeError("Could not find '## Takeaways' section in dev_ablation_results.md")
    current_results, takeaways = content.split(takeaways_marker, maxsplit=1)

    content = (
        current_results.rstrip()
        + EU_FULL_MARKER
        + sns_eu_full
        + casi_eu_full
        + mixed_eu_full
        + "\n## Takeaways\n"
        + takeaways.rstrip()
        + EU_SUMMARY_MARKER
        + sns_eu_summary
        + casi_eu_summary
        + mixed_eu_summary
        + "\n"
    )

    SUMMARY_PATH.write_text(content, encoding="utf-8")
    print("dev_ablation_results.md updated with Basque (EU) results.")


if __name__ == "__main__":
    main()
