"""
Appends CasiMedicos-dev and mixed-dev result sections to dev_ablation_results.md.
Run after eval_casi_mixed.sh completes.
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
    build_rows,
    expand_feedback_rows,
    fmt as fmt_full,
    make_row,
    td,
    th,
    write_markdown,
)

METRICS_DIR = Path("reports/metrics")
SUMMARY_PATH = Path("reports/metrics/es_dev_ablation_results.md")

# ── experiment definitions ────────────────────────────────────────────────────

CASI_EXPS = {
    0:  "39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev",
    1:  "40_mistral7b_rag_no_think_e5_topk1_extractive_sf_casimedicos_dev",
    2:  "41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev",
    3:  "42_mistral7b_rag_no_think_e5_topk5_extractive_sf_casimedicos_dev",
    4:  "43_mistral7b_rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev",
    5:  "44_mistral7b_rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev",
    6:  "45_mistral7b_rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev",
    7:  "46_mistral7b_random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev",
    8:  "47_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev",
    9:  "48_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev",
    10: "49_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev",
}

MIXED_EXPS = {
    0:  "50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev",
    1:  "51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev",
    2:  "52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev",
    3:  "53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev",
    4:  "54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev",
    5:  "55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev",
    6:  "56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev",
    7:  "57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev",
    8:  "58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev",
    9:  "59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev",
    10: "60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev",
}

CASI_LABELS = {
    0: "Baseline LLM only",
    1: "e5 top 1 (CasiMedicos index)",
    2: "e5 top 3 (CasiMedicos index)",
    3: "e5 top 5 (CasiMedicos index)",
    4: "rerank top 1 (CasiMedicos index)",
    5: "rerank top 3 (CasiMedicos index)",
    6: "rerank top 5 (CasiMedicos index)",
    7: "3-shot, no RAG",
    8: "3-shot + best rerank (CasiMedicos index)",
    9: "Cross-domain: SNS-only index",
    10: "Mixed-domain: SNS+CasiMedicos index",
}

MIXED_LABELS = {
    0: "Baseline LLM only",
    1: "e5 top 1 (mixed index)",
    2: "e5 top 3 (mixed index)",
    3: "e5 top 5 (mixed index)",
    4: "rerank top 1 (mixed index)",
    5: "rerank top 3 (mixed index)",
    6: "rerank top 5 (mixed index)",
    7: "3-shot, no RAG",
    8: "3-shot + best rerank (mixed index)",
    9: "Cross-domain: SNS-only index",
    10: "Cross-domain: CasiMedicos-only index",
}


def load(run_name: str) -> dict:
    return json.loads((METRICS_DIR / f"{run_name}.json").read_text())["summary"]


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
    """Return the ablation index (1-6) with highest overall BERTScore (noSF)."""
    best_idx, best_val = -1, -1.0
    for idx in range(1, 7):
        run = exps[idx]
        path = METRICS_DIR / f"{run}.json"
        if not path.exists():
            continue
        val = json.loads(path.read_text())["summary"]["before_feedback"]["overall"]["bertscore_f1"]
        if val > best_val:
            best_val, best_idx = val, idx
    return best_idx


def best_model_exp(exps: dict) -> int:
    """Return the experiment index with highest overall BERTScore (noSF or SF)."""
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


def build_section(title: str, dev_desc: str, exps: dict, labels: dict) -> str:
    rows = []
    best_idx = best_model_exp(exps)
    for idx in range(11):
        run = exps[idx]
        path = METRICS_DIR / f"{run}.json"
        if not path.exists():
            rows.append(f"| {idx} | {labels[idx]} | missing | missing |")
            continue
        d = load(run)
        bert = d["before_feedback"]["overall"]["bertscore_f1"]
        delta = d["self_feedback_delta"]["overall"]["bertscore_f1"]
        rows.append(summary_row(idx, labels[idx], fmt(bert), sign(delta), best_idx))

    best_idx = best_rag_exp(exps)
    best_label = f"exp {best_idx} ({parenthetical_label(labels[best_idx])})" if best_idx >= 0 else "exp 6 (assumed)"

    grouped_rows = [
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
    row_block = "\n".join(grouped_rows)

    return f"""
### {title}

*Evaluation set: {dev_desc}. Best RAG config from exps 1–6: {best_label}.*

| # | Experiment | BERT | SF Δ |
|---:|---|---:|---:|
{row_block}
"""


def build_full_section(title: str, dev_desc: str, exps: dict, labels: dict) -> str:
    best_idx = best_rag_exp(exps)
    best_note = (f"Best RAG config (exps 1–6): **exp {best_idx}** ({parenthetical_label(labels[best_idx])})."
                 if best_idx >= 0 else "Best RAG config pending.")

    status_rows = []
    for idx in range(11):
        run = exps[idx]
        status_rows.append(
            make_row(
                row_id=str(idx),
                experiment=labels[idx],
                status="done" if (METRICS_DIR / f"{run}.json").exists() else "missing",
                metrics_file=f"{run}.json",
                completed_run=run,
                notes="",
                metrics_dir=METRICS_DIR,
            )
        )

    result_rows = expand_feedback_rows(status_rows)
    table = build_metric_table(result_rows)

    return f"""
### {title}

*{dev_desc}. {best_note}*

All quality scores are out of 100. Cost is experiment-level and is reported per sample.

{table}
"""


def build_metric_table(result_rows: list[dict]) -> str:
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
        cells.extend(
            [
                td(fmt_full(row["mean_example_seconds"], signed=signed), strong_left=True),
                td(fmt_full(row["mean_input_tokens"], signed=signed), strong_left=True),
                td(fmt_full(row["mean_output_tokens"], signed=signed)),
                td(fmt_full(row["mean_total_tokens"], signed=signed)),
            ]
        )
        row_style = ' style="background-color: #fff8c5; font-weight: 700;"' if is_best else ""
        table_rows.append(f"    <tr{row_style}>" + "".join(cells) + "</tr>")

    return "\n".join(
        [
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
    )


KEY_INSIGHTS = """**Key insights:**

1. **RAG helps on all three dev sets, but the best configuration changes by task.** On SNS1064 dev, the best overall BERTScore is 3-shot + rerank top5 (76.72). On CasiMedicos dev, direct e5 top3 is best (79.41). On the mixed dev set, 3-shot + rerank top5 is best (77.59).
2. **Reranking is most useful for open-answer and mixed evaluation.** Rerank top5 is best among retrieval-only settings on SNS1064 (76.20) and mixed dev (77.21), while CasiMedicos multiple-choice prefers direct e5 top3 (79.41).
3. **Few-shot examples help when combined with retrieval, not alone.** 3-shot without RAG is weaker than the best RAG setup on all three dev sets, but 3-shot + rerank top5 gives the best score on SNS1064 and mixed dev.
4. **Cross-domain retrieval is asymmetric.** CasiMedicos-only retrieval performs poorly on SNS1064 (64.20) and mixed dev (69.31), while SNS-only retrieval is fairly competitive on CasiMedicos dev (75.92) and mixed dev (76.11), probably because the mixed dev set contains many SNS examples.
5. **Self-feedback is not a reliable quality booster.** Its BERTScore deltas are small and mixed across datasets, so given its computational cost (+1,632 tokens/sample on average) and temporal cost (+3.54 seconds/sample on average), it is not worth it.
"""


def main() -> None:
    sns_rows = build_rows(METRICS_DIR)
    write_markdown(sns_rows, expand_feedback_rows(sns_rows), SUMMARY_PATH)
    content = SUMMARY_PATH.read_text()

    casi_section = build_full_section(
        "CasiMedicos dev results",
        "CasiMedicos dev set, 55 examples, multiple-choice task",
        CASI_EXPS,
        CASI_LABELS,
    )
    mixed_section = build_full_section(
        "SNS1064+CasiMedicos dev results",
        "SNS1064+CasiMedicos dev set, 161 examples, mixed open-answer + multiple-choice",
        MIXED_EXPS,
        MIXED_LABELS,
    )

    casi_summary = build_section(
        "CasiMedicos dev summary",
        "CasiMedicos dev set, 55 examples, multiple-choice task",
        CASI_EXPS,
        CASI_LABELS,
    )
    mixed_summary = build_section(
        "SNS1064+CasiMedicos dev summary",
        "SNS1064+CasiMedicos dev set, 161 examples, mixed open-answer + multiple-choice",
        MIXED_EXPS,
        MIXED_LABELS,
    )

    takeaways_marker = "\n## Takeaways\n"
    if takeaways_marker not in content:
        raise RuntimeError("Could not find Takeaways section in es_dev_ablation_results.md")
    current_results, takeaways = content.split(takeaways_marker, maxsplit=1)

    content = (
        current_results.rstrip()
        + "\n"
        + casi_section
        + mixed_section
        + "\n## Takeaways\n"
        + takeaways.rstrip()
        + "\n"
        + casi_summary
        + mixed_summary
        + "\n"
        + KEY_INSIGHTS
    )
    SUMMARY_PATH.write_text(content)
    print("es_dev_ablation_results.md updated.")


if __name__ == "__main__":
    main()
