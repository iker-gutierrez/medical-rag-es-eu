"""Regenerate the ## Current results section of es_dev_ablation_results.md
with Mistral-7B (a) and Qwen3.5-9B (b) rows interspersed, numbered 0a/0b, 1a/1b...

Row structure per experiment:
  Xa  Mistral-7B-Instruct   no_think  noSF   (before_feedback metrics)
      Mistral-7B-Instruct   no_think  SF     (after_feedback metrics)
      Mistral-7B-Instruct   no_think  Δ      (delta)
  Xb  Qwen3.5-9B            no_think  noSF
      Qwen3.5-9B            no_think  SF
      Qwen3.5-9B            think     noSF
      Qwen3.5-9B            think     SF

think column: always no_think for Mistral; no_think/think for Qwen.
SF    column: noSF/SF/Δ for Mistral; noSF/SF for Qwen (no Δ row).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from write_supervisor_summary import (  # noqa: E402
    QUALITY_METRICS,
    QUALITY_METRIC_LABELS,
    SECTION_LABELS,
    SECTIONS,
    fmt,
    no_sf_metrics,
    delta_metrics,
    no_sf_cost,
    sf_cost,
    delta_cost,
    td,
    th,
)

METRICS_DIR = Path("reports/metrics")
SUMMARY_PATH = Path("reports/metrics/es_dev_ablation_results.md")

# ── Mistral run catalogue ─────────────────────────────────────────────────────

MISTRAL_RUNS = {
    "sns1064": [
        ("Baseline LLM only",        "17_mistral7b_no_rag_no_think_extractive_sf_dev"),
        ("e5 top 1",                 "32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev"),
        ("e5 top 3",                 "30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev"),
        ("e5 top 5",                 "33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev"),
        ("rerank top 1",             "28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev"),
        ("rerank top 3",             "31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev"),
        ("rerank top 5",             "29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev"),
        ("3-shot, no RAG",           "20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev"),
        ("3-shot + rerank top 5",    "38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev"),
        ("cross-domain retrieval",   "36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev"),
        ("mixed-domain retrieval",   "37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev"),
    ],
    "casimedicos": [
        ("Baseline LLM only",        "39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev"),
        ("e5 top 1",                 "40_mistral7b_rag_no_think_e5_topk1_extractive_sf_casimedicos_dev"),
        ("e5 top 3",                 "41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev"),
        ("e5 top 5",                 "42_mistral7b_rag_no_think_e5_topk5_extractive_sf_casimedicos_dev"),
        ("rerank top 1",             "43_mistral7b_rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev"),
        ("rerank top 3",             "44_mistral7b_rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev"),
        ("rerank top 5",             "45_mistral7b_rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev"),
        ("3-shot, no RAG",           "46_mistral7b_random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev"),
        ("3-shot + rerank top 5",    "47_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev"),
        ("cross-domain retrieval",   "48_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev"),
        ("mixed-domain retrieval",   "49_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev"),
    ],
    "mixed": [
        ("Baseline LLM only",        "50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev"),
        ("e5 top 1",                 "51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev"),
        ("e5 top 3",                 "52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev"),
        ("e5 top 5",                 "53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev"),
        ("rerank top 1",             "54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev"),
        ("rerank top 3",             "55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev"),
        ("rerank top 5",             "56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev"),
        ("3-shot, no RAG",           "57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev"),
        ("3-shot + rerank top 5",    "58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev"),
        ("cross-domain retrieval",   "59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev"),
        ("mixed-domain retrieval",   "60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev"),
    ],
}

# ── Qwen3.5-9B run catalogue ─────────────────────────────────────────────────

QWEN_EXP_LABELS = [
    "Baseline LLM only",
    "e5 top 1",
    "e5 top 3",
    "e5 top 5",
    "rerank top 1",
    "rerank top 3",
    "rerank top 5",
    "3-shot, no RAG",
    "3-shot + rerank top 5",
    "cross-domain retrieval",
    "mixed-domain retrieval",
]

# Slugs in display order (0–10). The config numeric ID = nosf_start + QWEN_CONFIG_IDX[i].
# Configs were generated in a different order: cross=8, mixed=9, 3shot=10.
# Display order puts 3shot+rerank at 8, cross at 9, mixed at 10 — matching the EU table.
QWEN_SLUGS = {
    "sns1064": [
        "no_rag", "rag_e5_topk1", "rag_e5_topk3", "rag_e5_topk5",
        "rag_e5_rerank1", "rag_e5_rerank3", "rag_e5_rerank5",
        "3shot_no_rag", "rag_3shot_e5_rerank5",
        "rag_cross_domain_e5_rerank5", "rag_mixed_e5_rerank5",
    ],
    "casimedicos": [
        "no_rag", "rag_e5_topk1", "rag_e5_topk3", "rag_e5_topk5",
        "rag_e5_rerank1", "rag_e5_rerank3", "rag_e5_rerank5",
        "3shot_no_rag", "rag_3shot_e5_rerank5",
        "rag_cross_domain_e5_rerank5", "rag_mixed_e5_rerank5",
    ],
    "mixed": [
        "no_rag", "rag_e5_topk1", "rag_e5_topk3", "rag_e5_topk5",
        "rag_e5_rerank1", "rag_e5_rerank3", "rag_e5_rerank5",
        "3shot_no_rag", "rag_3shot_e5_rerank5",
        "rag_sns1064_e5_rerank5", "rag_casimedicos_e5_rerank5",
    ],
}

# Maps display position → config offset within the 11-slot block.
# Configs generated: 0–7 same, then cross=8, mixed=9, 3shot=10.
# Display order: 0–7 same, then 3shot=8→config10, cross=9→config8, mixed=10→config9.
QWEN_CONFIG_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 9]

# (dev_slug, nosf_start_id, sf_start_id)
QWEN_DEV_IDS = {
    "sns1064":     (316, 382),
    "casimedicos": (338, 404),
    "mixed":       (360, 426),
}

THINK_OFFSET = 11  # think configs follow no_think configs within the 22-slot block

# ── metric loaders ────────────────────────────────────────────────────────────

def load_summary(run: str) -> dict | None:
    p = METRICS_DIR / f"{run}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def after_feedback_metrics(summary: dict | None, section: str) -> dict:
    if not summary:
        return {}
    return summary.get("after_feedback", {}).get(section, {})


def after_feedback_cost(summary: dict | None) -> dict:
    empty = {"mean_example_seconds": None, "mean_input_tokens": None,
             "mean_output_tokens": None, "mean_total_tokens": None}
    if not summary:
        return empty
    cost = summary.get("cost", {})
    timing = cost.get("timing", {})
    toks = cost.get("token_counts", {})
    inp = float(toks.get("input_tokens", {}).get("mean") or 0) + \
          float(toks.get("feedback_input_tokens", {}).get("mean") or 0)
    out = float(toks.get("output_tokens", {}).get("mean") or 0)
    total = float(toks.get("total_tokens", {}).get("mean") or 0) or inp + out
    secs = float(timing.get("example_seconds", {}).get("mean") or 0)
    return {"mean_example_seconds": secs, "mean_input_tokens": inp,
            "mean_output_tokens": out, "mean_total_tokens": total}

# ── row builders ──────────────────────────────────────────────────────────────

def _blank_metrics() -> dict:
    row: dict = {}
    for sec in SECTIONS:
        for metric in QUALITY_METRICS:
            row[f"{sec}_{metric}"] = None
    row.update({"mean_example_seconds": None, "mean_input_tokens": None,
                "mean_output_tokens": None, "mean_total_tokens": None})
    return row


def mistral_rows(exp_idx: int, label: str, run: str) -> list[dict]:
    """Return noSF / SF / Δ dicts for one Mistral experiment."""
    summary = load_summary(run)
    rows = []
    for condition in ("noSF", "SF", "Δ"):
        row: dict = {
            "exp_idx": exp_idx,
            "model": "Mistral-7B-Instruct",
            "experiment": label,
            "think": "no_think",
            "sf": condition,
            "_key": (exp_idx, "a", condition),
        }
        if summary is None:
            row.update(_blank_metrics())
        elif condition == "noSF":
            for sec in SECTIONS:
                sm = no_sf_metrics(summary, sec)
                for metric in QUALITY_METRICS:
                    row[f"{sec}_{metric}"] = sm.get(metric)
            row.update(no_sf_cost(summary))
        elif condition == "SF":
            for sec in SECTIONS:
                sm = after_feedback_metrics(summary, sec)
                for metric in QUALITY_METRICS:
                    row[f"{sec}_{metric}"] = sm.get(metric)
            row.update(after_feedback_cost(summary))
        else:  # Δ
            for sec in SECTIONS:
                dm = delta_metrics(summary, sec)
                for metric in QUALITY_METRICS:
                    row[f"{sec}_{metric}"] = dm.get(metric)
            row.update(delta_cost(summary))
        rows.append(row)
    return rows


def qwen_rows(exp_idx: int, label: str, dev_slug: str, exp_slug: str,
              nosf_start: int, sf_start: int, i: int) -> list[dict]:
    """Return 4 dicts (no_think/noSF, no_think/SF, think/noSF, think/SF) for one Qwen experiment."""
    rows = []
    for think in (False, True):
        offset = THINK_OFFSET if think else 0
        think_label = "think" if think else "no_think"
        mode = "think" if think else "no_think"

        config_i = QWEN_CONFIG_IDX[i]
        nosf_id = nosf_start + offset + config_i
        sf_id   = sf_start   + offset + config_i
        nosf_run = f"{nosf_id}_qwen35_9b_{exp_slug}_{mode}_extractive_{dev_slug}_dev"
        sf_run_  = f"{sf_id}_qwen35_9b_{exp_slug}_{mode}_extractive_{dev_slug}_sf_dev"

        for use_sf, run in ((False, nosf_run), (True, sf_run_)):
            sf_label = "SF" if use_sf else "noSF"
            summary = load_summary(run)
            row: dict = {
                "exp_idx": exp_idx,
                "model": "Qwen3.5-9B",
                "experiment": label,
                "think": think_label,
                "sf": sf_label,
                "_key": (exp_idx, "b", think_label, sf_label),
            }
            if summary is None:
                row.update(_blank_metrics())
            elif use_sf:
                for sec in SECTIONS:
                    sm = after_feedback_metrics(summary, sec)
                    for metric in QUALITY_METRICS:
                        row[f"{sec}_{metric}"] = sm.get(metric)
                row.update(after_feedback_cost(summary))
            else:
                for sec in SECTIONS:
                    sm = no_sf_metrics(summary, sec)
                    for metric in QUALITY_METRICS:
                        row[f"{sec}_{metric}"] = sm.get(metric)
                row.update(no_sf_cost(summary))
            rows.append(row)
    return rows

# ── table builder ─────────────────────────────────────────────────────────────

def _best_key_for(all_rows: list[dict], model_filter: str | None = None) -> tuple | None:
    best, best_score = None, -1.0
    for r in all_rows:
        if r.get("sf") == "Δ":
            continue
        if model_filter and r["model"] != model_filter:
            continue
        s = r.get("overall_bertscore_f1")
        if s is None:
            continue
        if float(s) > best_score:
            best_score = float(s)
            best = r["_key"]
    return best


def build_table(all_rows: list[dict]) -> str:
    quality_span = len(QUALITY_METRICS) * len(SECTIONS)
    header_1 = (
        th("#", rowspan=3)
        + th("model", rowspan=3)
        + th("experiment", rowspan=3)
        + th("think", rowspan=3)
        + th("SF", rowspan=3)
        + th("Quality ↑", colspan=quality_span, strong_left=True)
        + th("Cost ↓", colspan=4, strong_left=True)
    )
    header_2 = "".join(
        th(QUALITY_METRIC_LABELS[m], colspan=len(SECTIONS), strong_left=True)
        for m in QUALITY_METRICS
    )
    header_2 += th("sec/sample", rowspan=2, strong_left=True)
    header_2 += th("tokens/sample", colspan=3, strong_left=True)
    header_3 = ""
    for _m in QUALITY_METRICS:
        for idx, sec in enumerate(SECTIONS):
            header_3 += th(SECTION_LABELS[sec], strong_left=idx == 0)
    header_3 += th("input", strong_left=True) + th("output") + th("total")

    best_mistral = _best_key_for(all_rows, "Mistral-7B-Instruct")
    best_qwen    = _best_key_for(all_rows, "Qwen3.5-9B")
    best_overall = _best_key_for(all_rows)

    # Track row labels: emit Xa on first Mistral row of exp, Xb on first Qwen row
    exp_model_seen: set[tuple] = set()
    exp_num = -1
    prev_exp_idx: int | None = None

    table_rows = []
    for row in all_rows:
        exp_idx = row["exp_idx"]
        model_suffix = "a" if row["model"] == "Mistral-7B-Instruct" else "b"

        if exp_idx != prev_exp_idx:
            exp_num += 1
            prev_exp_idx = exp_idx

        key = (exp_idx, model_suffix)
        if key not in exp_model_seen:
            row_label = f"{exp_num}{model_suffix}"
            exp_model_seen.add(key)
        else:
            row_label = ""

        signed = row.get("sf") == "Δ"
        rk = row["_key"]
        is_best_mistral = rk == best_mistral
        is_best_qwen    = rk == best_qwen
        is_best_overall = rk == best_overall

        if is_best_overall:
            row_label = f"☞ {row_label}" if row_label else "☞"

        cells = [
            td(row_label),
            td(row["model"]),
            td(row["experiment"]),
            td(row["think"]),
            td(row["sf"]),
        ]
        for metric in QUALITY_METRICS:
            for idx, sec in enumerate(SECTIONS):
                cells.append(td(fmt(row[f"{sec}_{metric}"], signed=signed), strong_left=idx == 0))
        cells.extend([
            td(fmt(row["mean_example_seconds"], signed=signed), strong_left=True),
            td(fmt(row["mean_input_tokens"], signed=signed), strong_left=True),
            td(fmt(row["mean_output_tokens"], signed=signed)),
            td(fmt(row["mean_total_tokens"], signed=signed)),
        ])
        is_highlighted = is_best_mistral or is_best_qwen
        row_style = ' style="background-color: #fff8c5; font-weight: 700;"' if is_highlighted else ""
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


def build_dev_section(dev_slug: str) -> str:
    titles = {
        "sns1064":     ("SNS1064 dev results",
                        "SNS1064 dev set, 106 examples, open-answer task."),
        "casimedicos": ("CasiMedicos dev results",
                        "CasiMedicos dev set, 55 examples, multiple-choice task."),
        "mixed":       ("SNS1064+CasiMedicos dev results",
                        "SNS1064+CasiMedicos dev set, 161 examples, mixed open-answer + multiple-choice."),
    }
    title, desc = titles[dev_slug]
    nosf_start, sf_start = QWEN_DEV_IDS[dev_slug]

    all_rows: list[dict] = []
    mistral_entries = MISTRAL_RUNS[dev_slug]
    qwen_slugs = QWEN_SLUGS[dev_slug]

    for i, ((m_label, m_run), q_slug, q_label) in enumerate(
            zip(mistral_entries, qwen_slugs, QWEN_EXP_LABELS)):
        all_rows.extend(mistral_rows(i, m_label, m_run))
        all_rows.extend(qwen_rows(i, q_label, dev_slug, q_slug, nosf_start, sf_start, i))

    table = build_table(all_rows)
    return (
        f"\n### {title}\n\n"
        f"*{desc} "
        f"a = Mistral-7B-Instruct (noSF/SF/Δ), "
        f"b = Qwen3.5-9B (think × SF grid).*\n\n"
        f"All quality scores are out of 100. Cost is reported per sample.\n\n"
        f"{table}\n"
    )

# ── file patcher ──────────────────────────────────────────────────────────────

RESULTS_MARKER  = "\n## Current results\n"
TAKEAWAYS_MARKER = "\n## Takeaways\n"


def main() -> None:
    content = SUMMARY_PATH.read_text()

    if RESULTS_MARKER not in content:
        raise RuntimeError("Could not find '## Current results' marker")
    if TAKEAWAYS_MARKER not in content:
        raise RuntimeError("Could not find '## Takeaways' marker")

    before_results = content[:content.index(RESULTS_MARKER)]
    takeaways_onwards = content[content.index(TAKEAWAYS_MARKER):]

    new_results = RESULTS_MARKER
    for dev_slug in ("sns1064", "casimedicos", "mixed"):
        new_results += build_dev_section(dev_slug)

    SUMMARY_PATH.write_text(before_results + new_results + takeaways_onwards)
    print("es_dev_ablation_results.md regenerated with interspersed Mistral/Qwen3.5-9B rows.")


if __name__ == "__main__":
    main()
