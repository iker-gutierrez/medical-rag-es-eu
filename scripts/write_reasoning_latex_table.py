#!/usr/bin/env python
"""LaTeX tables for the reasoning pipelines, in the same format as the ablation
tables (scripts/write_result_tables.py): longtable, a two-level header (Quality
spanning ROUGE-L/BERT-F1/MC-acc/MeanQ, Cost spanning sec/tok/calls), and the
best value per metric column in bold.

Each pipeline is compared against the single-pass RAG baseline it was built on,
with retrieval held fixed, so any difference is attributable to the reasoning
procedure rather than to a change in the evidence supplied.

The baseline shows only its better-MeanQ SF state (noSF wins for both languages
currently), not both, matching the ablation tables' reference-row convention --
so there is no per-row SF split left to show, and no SF column: the baseline's
own row label ("Single-pass RAG: <model>, <config>") names the frozen
configuration directly rather than requiring the caption or a footnote to spell
it out.

`calls` (mean LLM generations per answer) is always a real column, in both
languages, not folded into a footnote: a three-round agentic loop and a
single-pass baseline both emit exactly one final answer, so cost reported
without it would make them look equally expensive.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "reports" / "metrics"
RUNS = REPO / "experiments" / "runs"
OUT = REPO / "manuscript"
SEEDS = [42, 43, 44]

QUALITY = [
    ("rouge_l_f1", "ROUGE-L"),
    ("bertscore_f1", "BERT-F1"),
    ("mc_accuracy", "MC-acc"),
    ("meanq", "MeanQ"),
]

# (display label, run stem, has self-feedback). ES uses Qwen3.5-9B no-think +
# rerank5 (think mode's MeanQ edge was 0.01, inside noise, at ~3x the cost); EU
# uses Latxa + retrieve top-3 -- Latxa's own ablation winner on the 2026-07-21
# staged rerun (rebuilt CasiMedicos-Exp/SNS-1064 splits + rebuilt retrieval
# indices, best_by_meanq_robust's variance/cost-aware pick: see
# scripts/write_result_tables.py's FORCED_REFERENCES comment for the full
# numbers and the cost-tiebreak rationale), and Llama's reasoning-pipeline base
# is ALSO retrieve top-3 on this rerun (Llama's own outright MeanQ winner, no
# tiebreak needed -- same comment). Both EU models now share one retrieval
# depth, so the EU pipeline rows (all built on Llama configs 1500-1503) are
# directly comparable to the single Latxa-based baseline row on retrieval
# depth, even though the baseline itself is only run once, on Latxa.
#
# The baseline's row label names its model + config directly and compactly
# ("Single-pass RAG: <model>, <config>") rather than the generic "Single-pass
# RAG (baseline)" -- a longer, more explicit inline label ("...(baseline: Qwen3.5-9B
# (no-think), rerank top 5)") was tried first and overflowed the page width,
# clipping the rightmost column; this compact form fits within the same column
# width the other rows use.
ES_BASELINE_DESC = "Qwen no-think, rerank top 5"
EU_BASELINE_DESC = "Latxa, retrieve top 3"
ES_ROWS = [
    (f"Single-pass RAG: {ES_BASELINE_DESC}", "1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev", True),
    # Structured CoT is reported in two retrieval variants: with our own frozen
    # best RAG config (same evidence as the baseline and the other three
    # pipelines -- isolates the effect of the four-stage causal reasoning
    # alone), and with MedCoT-RAG's own causal-aware retrieval scoring
    # (sec:reasoning-pipelines, src/medical_rag_thesis/causal_scoring.py) --
    # faithful to the full original method, retrieval and generation together.
    ("Structured CoT, our retrieval", "1530_qwen35_9b_no_think_structured_cot_meanq_best_extractive_mixed_dev", False),
    ("Structured CoT (MedCoT-RAG)", "1341_qwen35_9b_structured_cot_causal_no_think_extractive_mixed_dev", False),
    ("Thought-driven retrieval", "1531_qwen35_9b_no_think_thought_rag_meanq_best_extractive_mixed_dev", False),
    ("Thought-driven, iterative", "1532_qwen35_9b_no_think_thought_rag_iter_meanq_best_extractive_mixed_dev", False),
    ("Multi-round agentic (MA-RAG)", "1533_qwen35_9b_no_think_marag_meanq_best_extractive_mixed_dev", False),
]

# Qwen think's own reasoning-pipeline family: same rerank-top-5 base as
# ES_ROWS above (its own stage-A winner too; see FORCED_REFERENCES["ES"] in
# write_result_tables.py), reported as its own table (tab:reasoning-es-think)
# rather than appended into ES_ROWS, for the same reason Llama's EU rows are
# separate from Latxa's: each table's baseline row and per-column bolding are
# computed within that table alone. No MedCoT-RAG (causal-scoring) run exists
# for Qwen think specifically, so that row is omitted here rather than reusing
# the no-think run under a different model's label.
ES_THINK_ROWS = [
    # Row 1 is the TRUE base the four pipeline rows below are actually built
    # on: config 1280 (3-shot + rerank top 5, MeanQ 73.51, row 8's winner),
    # not plain rerank top 5 (1276, 72.13/72.24) -- configs 1554-1557 were
    # explicitly chained onto 1280 (see their own rag_base_source field), so
    # showing 1276 here as "the frozen baseline" was comparing the pipelines
    # against a config they were never built on top of, understating the gap.
    ("Single-pass RAG: Qwen think, 3-shot + rerank top 5", "1280_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev", True),
    ("Structured CoT, our retrieval", "1554_qwen35_9b_think_structured_cot_fewshot_rerank5_extractive_mixed_dev", False),
    ("Thought-driven retrieval", "1555_qwen35_9b_think_thought_rag_fewshot_rerank5_extractive_mixed_dev", False),
    ("Thought-driven, iterative", "1556_qwen35_9b_think_thought_rag_iter_fewshot_rerank5_extractive_mixed_dev", False),
    ("Multi-round agentic (MA-RAG)", "1557_qwen35_9b_think_marag_fewshot_rerank5_extractive_mixed_dev", False),
]

# Run stems below are the 2026-07-21 staged-rerun pipelines (config ids
# 1500-1513, "*_meanq_best_*" -- generated by scripts/run_all_models_full.py,
# wired to each model's own fresh best_by_meanq_robust winner). The prior
# 1310s/1324s config family (pre-rebuild retrieval indices, and for Llama's
# thought_rag/thought_rag_iter specifically, only 2 of 3 seeds ever completed)
# is superseded by these and no longer referenced here.
EU_ROWS = [
    (f"Single-pass RAG: {EU_BASELINE_DESC}", "1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev", True),
    ("Structured CoT, our retrieval", "1510_latxa_structured_cot_meanq_best_extractive_mixed_eu_dev", False),
    ("Structured CoT (MedCoT-RAG)", "1343_latxa_llama31_8b_structured_cot_causal_extractive_mixed_eu_dev", False),
    ("Thought-driven retrieval", "1511_latxa_thought_rag_meanq_best_extractive_mixed_eu_dev", False),
    ("Thought-driven, iterative", "1512_latxa_thought_rag_iter_meanq_best_extractive_mixed_eu_dev", False),
    ("Multi-round agentic (MA-RAG)", "1513_latxa_marag_meanq_best_extractive_mixed_eu_dev", False),
]

# Llama's own reasoning-pipeline family: same retrieve-top-3 base and no
# self-feedback as EU_ROWS above (its stage-A winner too; see
# FORCED_REFERENCES["EU"] in write_result_tables.py), reported as its own
# table (tab:reasoning-eu-llama) rather than appended into EU_ROWS, since the
# table's baseline/highlight logic assumes one frozen reference row per table.
EU_LLAMA_ROWS = [
    ("Single-pass RAG: Llama, retrieve top 3", "1042_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev", True),
    ("Structured CoT, our retrieval", "1500_llama31_8b_structured_cot_meanq_best_extractive_mixed_eu_dev", False),
    ("Structured CoT (MedCoT-RAG)", "1342_llama31_8b_structured_cot_causal_extractive_mixed_eu_dev", False),
    ("Thought-driven retrieval", "1501_llama31_8b_thought_rag_meanq_best_extractive_mixed_eu_dev", False),
    ("Thought-driven, iterative", "1502_llama31_8b_thought_rag_iter_meanq_best_extractive_mixed_eu_dev", False),
    ("Multi-round agentic (MA-RAG)", "1503_llama31_8b_marag_meanq_best_extractive_mixed_eu_dev", False),
]


def summaries_for(stem: str, suffix: str) -> list[dict[str, Any]]:
    out = []
    for seed in SEEDS:
        path = METRICS / f"{stem}_seed{seed}{suffix}.json"
        if path.exists():
            out.append(json.loads(path.read_text()).get("summary", {}))
    return out


def mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _value(summary: dict, section: str, name: str, use_sf: bool) -> Optional[float]:
    block = "after_feedback" if use_sf else "before_feedback"
    value = (summary.get(block) or {}).get(section, {}).get(name)
    if value is None and not use_sf:
        value = (summary.get(section) or {}).get(name)
    return float(value) if value is not None else None


def metric(summaries: list[dict], name: str, use_sf: bool) -> tuple[Optional[float], Optional[float]]:
    vals = [v for s in summaries if (v := _value(s, "overall", name, use_sf)) is not None]
    return mean_std(vals)


def meanq_per_seed(summaries: list[dict], mc_summaries: list[dict], use_sf: bool) -> tuple[Optional[float], Optional[float]]:
    """MeanQ per seed (pairing ROUGE-L/BERT-F1/MC-acc from the same seed by index --
    summaries and mc_summaries must be built from the same seed-ordered stem list),
    then averaged. Same method as scripts/metric_tables.py's meanq_per_seed, so this
    is directly comparable to the ablation decision tables and MeanQ column."""
    per_seed = []
    for i in range(len(summaries)):
        rouge = _value(summaries[i], "overall", "rouge_l_f1", use_sf)
        bert = _value(summaries[i], "overall", "bertscore_f1", use_sf)
        mc = _value(mc_summaries[i], "overall", "mc_accuracy", use_sf) if i < len(mc_summaries) else None
        parts = [v for v in (rouge, bert, mc) if v is not None]
        if parts:
            per_seed.append(sum(parts) / len(parts))
    return mean_std(per_seed)


def _nested_mean(block: dict, name: str) -> float:
    value = (block.get(name) or {}).get("mean")
    return float(value) if value is not None else 0.0


def cost(summaries: list[dict], token: bool, *, use_sf: bool) -> Optional[float]:
    """noSF and SF cost are not the same number: the raw metric JSON stores one
    pipeline-wide total (`example_seconds`, `total_tokens`) that already includes
    the self-feedback pass -- that total IS the SF row's cost. The noSF row's cost
    is the pre-feedback components only, matching scripts/summarize_metrics.py's
    cost_rows() split (also mirrored in scripts/write_result_tables.py's cost()).
    """
    vals = []
    for summary in summaries:
        block = summary.get("cost")
        if not block:
            continue
        timing = block.get("timing") or {}
        tokens = block.get("token_counts") or {}
        if token:
            value = (
                _nested_mean(tokens, "total_tokens") if use_sf else
                _nested_mean(tokens, "input_tokens") + _nested_mean(tokens, "initial_output_tokens")
            )
        else:
            value = (
                _nested_mean(timing, "example_seconds") if use_sf else
                _nested_mean(timing, "retrieval_seconds")
                + _nested_mean(timing, "rerank_seconds")
                + _nested_mean(timing, "few_shot_seconds")
                + _nested_mean(timing, "prompt_seconds")
                + _nested_mean(timing, "generation_seconds")
            )
        vals.append(value)
    return sum(vals) / len(vals) if vals else None


def llm_calls(stem: str) -> Optional[float]:
    vals = []
    for seed in SEEDS:
        path = RUNS / f"{stem}_seed{seed}" / "predictions.meta.json"
        if path.exists():
            value = json.loads(path.read_text()).get("mean_llm_calls_per_record")
            if value is not None:
                vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def fmt(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "---"
    return f"{mean:.2f}{{\\tiny$\\pm${std:.2f}}}" if std else f"{mean:.2f}"


def esc(text: str) -> str:
    return (str(text).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


# Labels highlighted with the same blue-bar treatment as the baseline row, per
# language -- for a reason other than "this is the frozen config everything
# else is compared against" (that's is_baseline). Currently: "Structured CoT,
# our retrieval" (config/row 2) in Spanish, the reasoning pipeline with the
# actual highest MeanQ on fresh Qwen no-think numbers (71.11, vs MedCoT-RAG's
# 67.69 and the single-pass baseline's 69.15) -- worth calling out visually.
# MedCoT-RAG was highlighted here previously on stale numbers (72.79 vs 69.79)
# that predate the think-mode parsing fix and the rebuilt dataset/index;
# config 2 is the genuine winner on the current, verified data.
EXTRA_HIGHLIGHT: dict[str, set[str]] = {
    "ES": {"Structured CoT, our retrieval"},
}

# Languages where the baseline row's blue bar is suppressed even though
# is_baseline is True -- the row still keeps its dashed-rule separator and
# bolded MeanQ (it's still the frozen reference the other rows are compared
# against), just not the highlight, since the highlight in this table is
# reserved for calling out MedCoT-RAG's win rather than marking the reference.
SUPPRESS_BASELINE_HIGHLIGHT: set[str] = {"ES"}


def build(rows, lang: str, suffix: str, dev: str, *, label_slug: Optional[str] = None) -> str:
    # MC-acc (and hence MeanQ) on the mixed table comes from the CasiMedicos
    # subset, matching scripts/meanq.py -- undefined on an open-answer-only suffix.
    mc_suffix = "_casimedicos" if suffix == "" else (suffix if suffix == "_casimedicos" else None)

    gathered = []
    for label, stem, has_sf in rows:
        summaries = summaries_for(stem, suffix)
        if not summaries:
            continue
        mc_summaries = summaries_for(stem, mc_suffix) if mc_suffix else []
        sf_states = (False, True) if has_sf else (False,)
        rows_this_label = []
        for use_sf in sf_states:
            row = {m: metric(summaries, m, use_sf) for m, _ in QUALITY if m != "meanq"}
            row["mc_accuracy"] = metric(mc_summaries, "mc_accuracy", use_sf) if mc_summaries else (None, None)
            row["meanq"] = meanq_per_seed(summaries, mc_summaries, use_sf)
            rows_this_label.append({
                "label": label,
                "sf": use_sf,
                "is_baseline": has_sf,
                "quality": row,
                "sec": cost(summaries, False, use_sf=use_sf),
                "tok": cost(summaries, True, use_sf=use_sf),
                "calls": llm_calls(stem),
            })
        if has_sf and len(rows_this_label) == 2:
            # The baseline is the frozen RAG config this whole table holds
            # fixed -- like a carried-forward reference row in the ablation
            # tables, only its better-MeanQ SF state is shown, not both. This
            # is also why there is no SF column: with every row (baseline
            # included) collapsed to a single state, and the reasoning
            # pipelines never having an SF pass at all, the column would be
            # blank everywhere it isn't simply redundant.
            nosf_mean = rows_this_label[0]["quality"]["meanq"][0]
            sf_mean = rows_this_label[1]["quality"]["meanq"][0]
            keep_sf = sf_mean is not None and (nosf_mean is None or sf_mean > nosf_mean)
            rows_this_label = [rows_this_label[1] if keep_sf else rows_this_label[0]]
        gathered.extend(rows_this_label)

    best = {}
    for m, _ in QUALITY:
        vals = [g["quality"][m][0] for g in gathered if g["quality"][m][0] is not None]
        if vals:
            best[m] = max(vals)

    # 2 label columns (#, Pipeline) + quality + cost (sec, tok, calls -- always
    # shown: a three-round agentic loop and a single-pass baseline both emit
    # exactly one final answer, so cost without it would make them look equally
    # expensive and conceal an order-of-magnitude compute difference).
    n_cost = 3
    ncol = 2 + len(QUALITY) + n_cost
    colspec = r"r l " + "c " * len(QUALITY) + "c " * n_cost
    header = (
        r"\# & Pipeline & \multicolumn{%d}{c}{Quality $\uparrow$} & "
        r"\multicolumn{%d}{c}{Cost $\downarrow$} \\" % (len(QUALITY), n_cost)
        + "\n" + r"\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}" % (
            2 + len(QUALITY), 3 + len(QUALITY), ncol)
        + "\n" + r" &  & " + " & ".join(n for _, n in QUALITY) + r" & sec & tok & calls \\"
    )
    lines = [
        r"\begin{scriptsize}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{" + colspec + r"}",
        r"\caption[Reasoning pipelines (%s)]{Reasoning pipelines on the frozen best "
        r"RAG configuration (%s, %s dev), row 1. Retrieval is held fixed, so every "
        r"difference is attributable to the reasoning procedure. Best value per "
        r"metric column in bold.%s} \label{tab:reasoning-%s} \\" % (
            lang, lang, dev,
            (r" MC-acc's mean$\pm$std can nominally exceed 100 near the ceiling "
             r"(row 3: two of three seeds score exactly 100), since the standard "
             r"deviation is a measure of seed-to-seed spread, not a bound on the "
             r"metric itself; no individual seed ever exceeds 100."
             if lang == "ES" else ""),
            label_slug or lang.lower()),
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{%d}{r}{\textit{continued on next page}} \\" % ncol,
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    extra_highlight = EXTRA_HIGHLIGHT.get(lang, set())
    for i, g in enumerate(gathered, start=1):
        # The baseline is the frozen RAG config every pipeline below it is
        # compared against -- highlighted the same way a carried-forward
        # reference row is in the ablation tables (scripts/write_result_tables.py):
        # a translucent blue row background, with its MeanQ bolded too.
        baseline_highlighted = g["is_baseline"] and lang not in SUPPRESS_BASELINE_HIGHLIGHT
        is_highlighted = baseline_highlighted or g["label"] in extra_highlight
        row_prefix = r"\rowcolor{pinnedrow}" if is_highlighted else ""
        cells = [str(i), esc(g["label"])]
        for m, _ in QUALITY:
            mean, std = g["quality"][m]
            cell = fmt(mean, std)
            is_col_best = mean is not None and m in best and mean == best[m]
            is_baseline_meanq = m == "meanq" and g["is_baseline"]
            if is_col_best or is_baseline_meanq:
                cell = r"\textbf{%s}" % cell
            cells.append(cell)
        cells += [
            f"{g['sec']:.2f}" if g["sec"] is not None else "---",
            f"{g['tok']:.0f}" if g["tok"] is not None else "---",
            f"{g['calls']:.1f}" if g["calls"] is not None else "1.0",
        ]
        lines.append(row_prefix + " & ".join(cells) + r" \\")
        # Dashed rule after the baseline row, splitting "the frozen config
        # everything else is compared against" from "the pipelines being
        # compared" -- same convention as the ablation tables' reference/
        # new-comparison split, with matching spacing on both sides.
        if g["is_baseline"]:
            lines.append(r"\addlinespace[4pt]")
            lines.append(r"\cdashline{1-%d}" % ncol)
            lines.append(r"\addlinespace[4pt]")

    lines += [
        r"\end{longtable}",
        r"\end{scriptsize}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    (OUT / "table_reasoning_es.tex").write_text(build(ES_ROWS, "ES", "", "mixed"))
    (OUT / "table_reasoning_es_think.tex").write_text(
        build(ES_THINK_ROWS, "ES, Qwen think", "", "mixed", label_slug="es-think")
    )
    (OUT / "table_reasoning_eu.tex").write_text(build(EU_ROWS, "EU", "", "mixed"))
    (OUT / "table_reasoning_eu_llama.tex").write_text(
        build(EU_LLAMA_ROWS, "EU, Llama", "", "mixed", label_slug="eu-llama")
    )
    print("  wrote table_reasoning_es.tex, table_reasoning_es_think.tex, "
          "table_reasoning_eu.tex, table_reasoning_eu_llama.tex")


if __name__ == "__main__":
    main()
