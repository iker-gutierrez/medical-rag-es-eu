#!/usr/bin/env python
"""
Replace the SNS1064+CasiMedicos section in es_dev_ablation_results.md with
new rows from the vLLM runs (configs 1128-1138 Qwen3.5-9B, 1260-1270 Mistral),
computing mean±std across seeds 42/43/44.

Table structure exactly matches the existing multi-model mixed table:
  #, model, experiment, think, SF, [8 metrics × 3 sections], sec/sample, input, output, total
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from write_supervisor_summary import (  # noqa: E402
    QUALITY_METRICS,
    QUALITY_METRIC_LABELS,
    SECTION_LABELS,
    SECTIONS,
    no_sf_metrics,
    no_sf_cost,
)

METRICS_DIR = ROOT / "reports" / "metrics"
REPORT_PATH = ROOT / "reports" / "metrics" / "es_dev_ablation_results.md"

SEEDS = [42, 43, 44]

EXPERIMENTS = [
    # (display_label, mistral_run_id_prefix, mistral_base, qwen_run_id_prefix, qwen_base,
    #  qwen_think_run_id_prefix, qwen_think_base)
    ("Baseline LLM only",              "1260", "mistral7b_no_rag_no_think_extractive_mixed_dev",              "1128", "qwen35_9b_no_rag_no_think_extractive_mixed_dev",
     "1270", "qwen35_9b_no_rag_think_extractive_mixed_dev"),
    ("retrieve top 1",                 "1261", "mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev",        "1129", "qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev",
     "1271", "qwen35_9b_rag_e5_topk1_think_extractive_mixed_dev"),
    ("retrieve top 3",                 "1262", "mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev",        "1130", "qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev",
     "1272", "qwen35_9b_rag_e5_topk3_think_extractive_mixed_dev"),
    ("retrieve top 5",                 "1263", "mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev",        "1131", "qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev",
     "1273", "qwen35_9b_rag_e5_topk5_think_extractive_mixed_dev"),
    ("rerank top 1",                   "1264", "mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev",      "1132", "qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev",
     "1274", "qwen35_9b_rag_e5_rerank1_think_extractive_mixed_dev"),
    ("rerank top 3",                   "1265", "mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev",      "1133", "qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev",
     "1275", "qwen35_9b_rag_e5_rerank3_think_extractive_mixed_dev"),
    ("rerank top 5",                   "1266", "mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev",      "1134", "qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev",
     "1276", "qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev"),
    ("3-shot, no RAG",                 "1267", "mistral7b_3shot_no_rag_no_think_extractive_mixed_dev",        "1135", "qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev",
     "1277", "qwen35_9b_3shot_no_rag_think_extractive_mixed_dev"),
    ("3-shot + rerank top 5",          "1270", "mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev","1138", "qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev",
     "1280", "qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev"),
    ("cross-domain: SNS index",        "1268", "mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev","1136","qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev",
     "1278", "qwen35_9b_rag_sns1064_e5_rerank5_think_extractive_mixed_dev"),
    ("cross-domain: CasiMedicos index","1269", "mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev","1137","qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev",
     "1279", "qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev"),
]

# The three ES models and which EXPERIMENTS tuple columns give their (id, base).
#   tuple = (label, mistral_id, mistral_base, qwen_id, qwen_base, qwenthink_id, qwenthink_base)
ES_MODELS = [
    ("Mistral-7B-Instruct",   1, 2),
    ("Qwen3.5-9B (no-think)", 3, 4),
    ("Qwen3.5-9B (think)",    5, 6),
]
# Retrieval sweep rows the base RAG config for row 8 / domain rows is chosen from
# (indices into EXPERIMENTS): 1 retrieve top1, 2 retrieve top3, 3 retrieve top5, 4 rerank1, 5 rerank3,
# 6 rerank5. Row 8 cannot wire to itself, so this stays the six-config sweep for
# that decision -- see rewire_dependent_configs.py.
RETRIEVAL_IDX = [1, 2, 3, 4, 5, 6]
# Full pool for the decision table: it answers "what is the single best config to
# carry into the reasoning pipelines", so it must span every row (0 baseline, 1-6
# retrieval sweep, 7 few-shot-no-RAG, 8 few-shot+RAG, 9-10 domain restriction), not
# just the retrieval sweep. Row 8 can genuinely win outright -- for Qwen-think it
# does, 71.35 vs. rerank5's 71.34 -- and the table must be able to show that.
DECISION_IDX = list(range(11))


def _load_summaries_for(source_suffix: str):
    def loader(id_prefix: str, base: str) -> list:
        return [load_summary(run_dir(id_prefix, base, s, source_suffix), source_suffix)
                for s in SEEDS]
    return loader


def build_meanq_decision_tables(source_suffix: str = "") -> str:
    """MeanQ-ranked decision table per model, over the retrieval sweep.

    Shows every metric (ROUGE-L, BERT-F1, MC-acc, MeanQ, sec/sample, tok/sample),
    each split noSF/SF/Δ, ranked by noSF MeanQ -- the score the staged ablation uses
    to pick the base RAG config that the few-shot and domain rows are wired to. Same
    per-seed MeanQ as scripts/meanq.py, so the ranking never drifts from selection.
    """
    from metric_tables import render_model_table  # noqa: E402
    include_mc = source_suffix != "_sns1064"
    loader = _load_summaries_for(source_suffix)

    parts = ["\n### Best RAG config by MeanQ (ES mixed dev task)\n",
             "MeanQ = mean(ROUGE-L, BERT-F1, MC-acc), per seed then averaged, on the "
             "no-self-feedback prediction; rows are ranked by it. This is the score "
             "the staged ablation uses to choose the base RAG config that the few-shot "
             "and domain-restriction experiments are then wired to.\n"]
    for label, id_col, base_col in ES_MODELS:
        rows = [(0, EXPERIMENTS[i][0], EXPERIMENTS[i][id_col], EXPERIMENTS[i][base_col])
                for i in DECISION_IDX]
        parts.append(f"\n#### {label}\n")
        parts.append(render_model_table(
            rows, cost_nosf=_cost_nosf, cost_sf=_cost_sf, cost_delta=_cost_delta,
            load_summaries=loader, include_mc=include_mc, rank_by_meanq=True))
    return "\n".join(parts)

THINK_VS_NOTHINK_NOTE = """
### Why Qwen think mode scores lower than no_think on this task

Across almost every RAG configuration, Qwen3.5-9B in think mode scores lower
than no_think mode, and the gap is much larger on ROUGE-L (lexical overlap)
than on BERTScore (semantic similarity) -- e.g. for "rerank top 5" (mixed
dev): BERTScore drops 78.22 -> 74.11 (~4 pts) while ROUGE-L drops
40.69 -> 28.96 (~12 pts, ~30% relative). This is the signature of a
**style mismatch, not a correctness problem**: think mode's answers are
often semantically reasonable but phrased in a more discursive, explanatory
register instead of the terse extractive style the reference answers use.

Concrete example (same record, same rerank-top-5 config, seed 42):

| | Answer |
|---|---|
| Reference | `Variable.` |
| no_think | `Variable.` |
| think | `No se identificaron estudios que respondieran a la pregunta.` |

The user prompt explicitly instructs "COPIA frases exactas cuando sea
posible" (copy exact phrases when possible) for RAG-context examples, and
no_think mode follows this closely -- it has no room to elaborate, since the
chat template pre-fills an empty, closed `<think>\\n\\n</think>` block before
generation starts. Think mode, by contrast, is given an open `<think>` block
it must fill with reasoning and close itself; after several paragraphs of
discursive reasoning, the model's final answer inherits that same register
rather than snapping back to terse copying, even though it received the
identical instruction and sampling hyperparameters as no_think mode.

This was verified to not be a prompt or hyperparameter difference: aside from
`think`/`max_new_tokens`/the vLLM `reasoning_parser`/`thinking_token_budget`
fields (all necessary infrastructure for enabling reasoning), the two configs
are otherwise identical. The effect appears to be a genuine behavioral
trade-off of chain-of-thought reasoning on terse extractive-QA tasks, worth
noting as a limitation of think mode for this dataset rather than an
implementation bug.
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def run_dir(id_prefix: str, base: str, seed: int, source_suffix: str = "") -> str:
    """Resolve a run directory. Every seed is explicit: `<id>_<base>_seed<N>`.

    The seeded re-run writes all three seeds under the explicit `_seed{N}` name, so
    that is the only form consulted. The two fallbacks that used to live here have
    been removed on purpose:

      * `_updated_seed{N}` -- an older regeneration. Kept as the FIRST choice, it
        would silently shadow the new runs wherever an old _updated metric survived,
        mixing pre-seed-fix and post-seed-fix numbers in one table.
      * bare `<id>_<base>` for seed 42 -- the old convention where seed 42 had no
        suffix. That made "the new suffixless run" and "the old seed-42 run" the same
        path, so a re-run would overwrite the archive in place.

    The superseded runs live in experiments/runs_v2 / reports/metrics_v2.
    """
    return f"{id_prefix}_{base}_seed{seed}"


def load_summary(run: str, suffix: str = "") -> dict | None:
    p = METRICS_DIR / f"{run}{suffix}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("summary")


def _vals(summaries: list[dict | None], section: str, metric: str,
          use_after: bool = False) -> list[float]:
    out = []
    for s in summaries:
        if not s:
            continue
        if use_after:
            v = s.get("after_feedback", {}).get(section, {}).get(metric)
        else:
            # before_feedback holds the initial (noSF) prediction metrics
            v = s.get("before_feedback", {}).get(section, {}).get(metric)
            if v is None:
                # fallback for runs without self-feedback
                v = s.get(section, {}).get(metric)
        if v is not None:
            out.append(float(v))
    return out


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    std = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, std


def fmt(v: float | None, *, signed: bool = False) -> str:
    if v is None:
        return ""
    s = f"{v:.2f}"
    return ("+" + s) if signed and v > 0 else s


def fmt_ms(m: float | None, s: float | None) -> str:
    """Format mean±std, both rounded to 2 dp. Always show ±."""
    if m is None:
        return ""
    return f"{m:.2f}±{s:.2f}" if s is not None else f"{m:.2f}"


def _sec_nosf(summary: dict | None) -> float | None:
    """Seconds per sample for the noSF (initial generation) phase only."""
    if not summary:
        return None
    t = summary.get("cost", {}).get("timing", {})
    # example_seconds includes feedback; reconstruct noSF portion
    phases = ["retrieval_seconds", "rerank_seconds", "few_shot_seconds",
              "prompt_seconds", "generation_seconds"]
    total = sum(float(t.get(p, {}).get("mean") or 0) for p in phases)
    return total if total > 0 else None


def _sec_sf(summary: dict | None) -> float | None:
    """Seconds per sample for the full SF run (initial + feedback generation)."""
    if not summary:
        return None
    t = summary.get("cost", {}).get("timing", {})
    v = t.get("example_seconds", {}).get("mean")
    return float(v) if v is not None else None


def _tok(summary: dict | None, key: str) -> float | None:
    if not summary:
        return None
    v = summary.get("cost", {}).get("token_counts", {}).get(key, {}).get("mean")
    return float(v) if v is not None else None


def _cost_nosf(summaries: list[dict | None]) -> dict[str, str]:
    """Aggregate cost fields for the noSF (initial generation) phase."""
    sec_vals = [_sec_nosf(s) for s in summaries]
    m, s = _mean_std([v for v in sec_vals if v is not None])
    row: dict[str, str] = {"sec_per_sample": fmt_ms(m, s)}

    in_vals  = [_tok(s, "input_tokens")         for s in summaries]
    out_vals = [_tok(s, "initial_output_tokens") for s in summaries]
    mi, si = _mean_std([v for v in in_vals  if v is not None])
    mo, so = _mean_std([v for v in out_vals if v is not None])
    row["mean_input_tokens"]  = fmt_ms(mi, si)
    row["mean_output_tokens"] = fmt_ms(mo, so)
    tot_vals = [i + o for i, o in zip(in_vals, out_vals) if i is not None and o is not None]
    mt, st = _mean_std(tot_vals)
    row["mean_total_tokens"] = fmt_ms(mt, st)
    return row


def _cost_sf(summaries: list[dict | None]) -> dict[str, str]:
    """Aggregate cost fields for the full SF run (initial + feedback)."""
    sec_vals = [_sec_sf(s) for s in summaries]
    m, s = _mean_std([v for v in sec_vals if v is not None])
    row: dict[str, str] = {"sec_per_sample": fmt_ms(m, s)}

    # input = first prompt + feedback prompt; output = final output tokens; total = all
    in1_vals  = [_tok(s, "input_tokens")          for s in summaries]
    in2_vals  = [_tok(s, "feedback_input_tokens") for s in summaries]
    out_vals  = [_tok(s, "output_tokens")         for s in summaries]
    tot_vals  = [_tok(s, "total_tokens")          for s in summaries]

    combined_in = [
        a + b for a, b in zip(in1_vals, in2_vals)
        if a is not None and b is not None
    ]
    mi, si = _mean_std(combined_in)
    mo, so = _mean_std([v for v in out_vals  if v is not None])
    mt, st = _mean_std([v for v in tot_vals  if v is not None])
    row["mean_input_tokens"]  = fmt_ms(mi, si)
    row["mean_output_tokens"] = fmt_ms(mo, so)
    row["mean_total_tokens"]  = fmt_ms(mt, st)
    return row


def _cost_delta(summaries: list[dict | None]) -> dict[str, str]:
    """SF − noSF cost delta (signed)."""
    def _delta_fmt(sf_vals, nosf_vals):
        pairs = [(a - b) for a, b in zip(sf_vals, nosf_vals)
                 if a is not None and b is not None]
        m, s = _mean_std(pairs)
        if m is None:
            return ""
        sign = "+" if m >= 0 else ""
        return f"{sign}{m:.2f}±{s:.2f}" if s is not None else f"{sign}{m:.2f}"

    sec_sf   = [_sec_sf(s)   for s in summaries]
    sec_nosf = [_sec_nosf(s) for s in summaries]
    row: dict[str, str] = {"sec_per_sample": _delta_fmt(sec_sf, sec_nosf)}

    # SF input = first prompt + feedback prompt; noSF input = first prompt only
    in1  = [_tok(s, "input_tokens")          for s in summaries]
    in2  = [_tok(s, "feedback_input_tokens") for s in summaries]
    combined_sf_in = [
        a + b for a, b in zip(in1, in2) if a is not None and b is not None
    ]
    out_sf   = [_tok(s, "output_tokens")          for s in summaries]
    out_nosf = [_tok(s, "initial_output_tokens")  for s in summaries]
    tot_sf   = [_tok(s, "total_tokens")           for s in summaries]
    tot_nosf = [i + o for i, o in zip(in1, out_nosf) if i is not None and o is not None]

    row["mean_input_tokens"]  = _delta_fmt(combined_sf_in, in1)
    row["mean_output_tokens"] = _delta_fmt(out_sf, out_nosf)
    row["mean_total_tokens"]  = _delta_fmt(tot_sf, tot_nosf)
    return row


def aggregate(id_prefix: str, base: str, suffix: str = "") -> dict[str, Any]:
    summaries = [load_summary(run_dir(id_prefix, base, s, suffix), suffix) for s in SEEDS]
    row: dict[str, Any] = {}
    for sec in SECTIONS:
        for met in QUALITY_METRICS:
            vals = _vals(summaries, sec, met, use_after=False)
            m, s = _mean_std(vals)
            row[f"{sec}_{met}"] = fmt_ms(m, s)
    row.update(_cost_nosf(summaries))
    row["n_seeds"] = len([x for x in summaries if x])
    return row


# ── table builder (mirrors update_es_ablation_with_qwen3.build_table) ─────────

def cell_style(*, strong_left: bool = False) -> str:
    style = "border-left: 1px solid #d0d7de; border-right: 1px solid #d0d7de; padding: 4px 6px;"
    if strong_left:
        style += " border-left: 2px solid #57606a;"
    return style


def th(content: str, *, colspan: int | None = None, rowspan: int | None = None,
       strong_left: bool = False) -> str:
    attrs = [f'style="{cell_style(strong_left=strong_left)}"']
    if colspan is not None:
        attrs.append(f'colspan="{colspan}"')
    if rowspan is not None:
        attrs.append(f'rowspan="{rowspan}"')
    return f"<th {' '.join(attrs)}>{html.escape(content)}</th>"


def td(content: Any, *, strong_left: bool = False) -> str:
    return f'<td style="{cell_style(strong_left=strong_left)}">{html.escape(str(content))}</td>'


def _best_per_model_think(all_rows: list[dict]) -> dict[tuple[str, str], str]:
    """Return {(model_label, think): experiment_label} for the experiment with the
    highest max(noSF, SF) BERTScore, computed separately per (model, think) group so
    Qwen no_think and Qwen think are tracked independently rather than merged."""
    scores: dict[tuple[str, str, str], float] = {}
    for row in all_rows:
        val_str = row.get("overall_bertscore_f1", "")
        if not val_str:
            continue
        key = (row["model"], row.get("think", ""), row.get("experiment", ""))
        mean_val = float(val_str.split("±")[0])
        scores[key] = max(scores.get(key, -1.0), mean_val)
    best: dict[tuple[str, str], tuple[float, str]] = {}
    for (model, think, exp), val in scores.items():
        group_key = (model, think)
        if group_key not in best or val > best[group_key][0]:
            best[group_key] = (val, exp)
    return {group_key: exp for group_key, (_, exp) in best.items()}


def build_table(all_rows: list[dict], *, include_mc_accuracy: bool = False) -> str:
    best_per_group_map = _best_per_model_think(all_rows)

    quality_span = len(QUALITY_METRICS) * len(SECTIONS) + (1 if include_mc_accuracy else 0)
    header_1 = (
        th("#", rowspan=3)
        + th("model", rowspan=3)
        + th("experiment", rowspan=3)
        + th("think", rowspan=3)
        + th("SF", rowspan=3)
        + th("Quality ↑", colspan=quality_span, strong_left=True)
        + th("Cost ↓", colspan=4, strong_left=True)
    )
    header_2 = ""
    if include_mc_accuracy:
        header_2 += th("MC Accuracy", rowspan=2, strong_left=True)
    header_2 += "".join(
        th(QUALITY_METRIC_LABELS[m], colspan=len(SECTIONS), strong_left=(idx == 0 and not include_mc_accuracy))
        for idx, m in enumerate(QUALITY_METRICS)
    )
    header_2 += th("sec/sample", rowspan=2, strong_left=True)
    header_2 += th("tokens/sample", colspan=3, strong_left=True)
    header_3 = ""
    for m_idx, _m in enumerate(QUALITY_METRICS):
        for idx, sec in enumerate(SECTIONS):
            strong = idx == 0 and not (include_mc_accuracy and m_idx == 0)
            header_3 += th(SECTION_LABELS[sec], strong_left=strong)
    header_3 += th("input", strong_left=True) + th("output") + th("total")

    table_rows = []
    for row in all_rows:
        label = row.get("label", "")
        model = row["model"]
        exp = row.get("experiment", "")
        sf = row.get("sf", "")
        think = row.get("think", "")
        # Highlight the winning experiment within each of the three groups:
        # Mistral, Qwen no_think, Qwen think -- tracked independently so each
        # gets its own best-row highlight rather than one global winner.
        is_best_in_group = best_per_group_map.get((model, think)) == exp
        row_style = ' style="background-color: #fff8c5;"' if is_best_in_group else ""
        cells = [
            td(label),
            td(model),
            td(exp),
            td(row["think"]),
            td(sf),
        ]
        if include_mc_accuracy:
            cells.append(td(row.get("mc_accuracy", ""), strong_left=True))
        for m_idx, metric in enumerate(QUALITY_METRICS):
            for idx, sec in enumerate(SECTIONS):
                strong = idx == 0 and not (include_mc_accuracy and m_idx == 0)
                cells.append(td(row.get(f"{sec}_{metric}", ""), strong_left=strong))
        cells.extend([
            td(row.get("sec_per_sample", ""), strong_left=True),
            td(row.get("mean_input_tokens", ""), strong_left=True),
            td(row.get("mean_output_tokens", "")),
            td(row.get("mean_total_tokens", "")),
        ])
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


# ── main ──────────────────────────────────────────────────────────────────────

def _emit_model_rows(
    all_rows: list[dict],
    summaries: list[dict | None],
    *,
    model_label: str,
    model_suffix: str,
    label: str,
    exp_num: int,
    think_label: str,
) -> None:
    """Append noSF, SF, and Δ rows for one (model, think-mode) combination."""
    cost_nosf = _cost_nosf(summaries)
    cost_sf   = _cost_sf(summaries)
    cost_delta = _cost_delta(summaries)

    for sf_label, use_after, cost in [
        ("noSF", False, cost_nosf),
        ("SF",   True,  cost_sf),
    ]:
        row: dict[str, Any] = {}
        for sec in SECTIONS:
            for met in QUALITY_METRICS:
                vals = _vals(summaries, sec, met, use_after=use_after)
                m_val, s_val = _mean_std(vals)
                row[f"{sec}_{met}"] = fmt_ms(m_val, s_val)
        # mc_accuracy: CasiMedicos-only, short_answer section, not part of the
        # generic per-section quality grid (see _vals + "overall" key which
        # evaluate_records already populates from the short_answer value).
        mc_vals = _vals(summaries, "overall", "mc_accuracy", use_after=use_after)
        mc_m, mc_s = _mean_std(mc_vals)
        row["mc_accuracy"] = fmt_ms(mc_m, mc_s)
        row.update(cost)
        row.update({
            "label": f"{exp_num}{model_suffix}" if sf_label == "noSF" else "",
            "model": model_label,
            "experiment": label,
            "think": think_label,
            "sf": sf_label,
        })
        all_rows.append(row)

    # Δ row: SF − noSF
    delta_row: dict[str, Any] = {}
    for sec in SECTIONS:
        for met in QUALITY_METRICS:
            nosf_vals = _vals(summaries, sec, met, use_after=False)
            sf_vals   = _vals(summaries, sec, met, use_after=True)
            pairs = [a - b for a, b in zip(sf_vals, nosf_vals)]
            m_val, s_val = _mean_std(pairs)
            if m_val is not None:
                sign = "+" if m_val >= 0 else ""
                delta_row[f"{sec}_{met}"] = (
                    f"{sign}{m_val:.2f}±{s_val:.2f}" if s_val is not None
                    else f"{sign}{m_val:.2f}"
                )
            else:
                delta_row[f"{sec}_{met}"] = ""
    mc_nosf_vals = _vals(summaries, "overall", "mc_accuracy", use_after=False)
    mc_sf_vals   = _vals(summaries, "overall", "mc_accuracy", use_after=True)
    mc_pairs = [a - b for a, b in zip(mc_sf_vals, mc_nosf_vals)]
    mc_dm, mc_ds = _mean_std(mc_pairs)
    if mc_dm is not None:
        sign = "+" if mc_dm >= 0 else ""
        delta_row["mc_accuracy"] = f"{sign}{mc_dm:.2f}±{mc_ds:.2f}" if mc_ds is not None else f"{sign}{mc_dm:.2f}"
    else:
        delta_row["mc_accuracy"] = ""
    delta_row.update(cost_delta)
    delta_row.update({
        "label": "",
        "model": model_label,
        "experiment": label,
        "think": think_label,
        "sf": "Δ",
    })
    all_rows.append(delta_row)


def build_rows_for_source(source_suffix: str) -> list[dict]:
    """Build all experiment rows for one source split ('' = mixed, '_sns1064', '_casimedicos')."""
    all_rows: list[dict] = []
    exp_num = 0
    for label, m_id, m_base, q_id, q_base, qt_id, qt_base in EXPERIMENTS:
        m_summaries = [load_summary(run_dir(m_id, m_base, s, source_suffix), source_suffix) for s in SEEDS]
        q_summaries = [load_summary(run_dir(q_id, q_base, s, source_suffix), source_suffix) for s in SEEDS]
        qt_summaries = [load_summary(run_dir(qt_id, qt_base, s, source_suffix), source_suffix) for s in SEEDS]

        if not any(m_summaries) and not any(q_summaries) and not any(qt_summaries):
            print(f"WARNING: no metrics for exp {exp_num} ({label}) suffix='{source_suffix}'")
            exp_num += 1
            continue

        _emit_model_rows(all_rows, m_summaries, model_label="Mistral-7B-Instruct",
                          model_suffix="a", label=label, exp_num=exp_num, think_label="no_think")
        _emit_model_rows(all_rows, q_summaries, model_label="Qwen3.5-9B",
                          model_suffix="b", label=label, exp_num=exp_num, think_label="no_think")
        if any(qt_summaries):
            _emit_model_rows(all_rows, qt_summaries, model_label="Qwen3.5-9B",
                              model_suffix="b", label=label, exp_num=exp_num, think_label="think")

        exp_num += 1
    return all_rows


def load_sf_delta(id_prefix: str, base: str, suffix: str) -> tuple[float | None, float | None]:
    deltas = []
    for seed in SEEDS:
        rd = run_dir(id_prefix, base, seed, suffix)
        p = METRICS_DIR / f"{rd}{suffix}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text()).get("summary", {})
        v = s.get("self_feedback_delta", {}).get("overall", {}).get("bertscore_f1")
        if v is not None:
            deltas.append(float(v))
    return _mean_std(deltas)


def build_summary_table_for_source(
    source_suffix: str,
    section_desc: str,
    best_rag_note: str,
) -> str:
    """Per-model summary table over ALL experiments, showing every metric
    (ROUGE-L, BERT-F1, MC-acc, MeanQ, sec/sample, tok/sample) split noSF/SF/Δ."""
    from metric_tables import render_model_table  # noqa: E402
    include_mc = source_suffix != "_sns1064"
    loader = _load_summaries_for(source_suffix)

    lines = [f"\n*{section_desc}*\n"]
    if best_rag_note:
        lines.append(f"*{best_rag_note}*\n")

    for model_label, id_col, base_col in ES_MODELS:
        rows = [(i, EXPERIMENTS[i][0], EXPERIMENTS[i][id_col], EXPERIMENTS[i][base_col])
                for i in range(len(EXPERIMENTS))]
        lines.append(f"\n**{model_label}**\n")
        lines.append(render_model_table(
            rows, cost_nosf=_cost_nosf, cost_sf=_cost_sf, cost_delta=_cost_delta,
            load_summaries=loader, include_mc=include_mc, rank_by_meanq=False))

    return "\n".join(lines) + "\n"


def make_section(title: str, subtitle: str, source_suffix: str) -> str:
    rows = build_rows_for_source(source_suffix)
    if not rows:
        return ""
    # mc_accuracy only makes sense where CasiMedicos records are present:
    # the CasiMedicos-only section and the mixed section, not SNS1064-only
    # (open-answer task, no multiple-choice concept).
    include_mc_accuracy = source_suffix != "_sns1064"
    table = build_table(rows, include_mc_accuracy=include_mc_accuracy)
    return (
        f"\n### {title}\n\n"
        f"*{subtitle}*\n\n"
        "All quality scores are out of 100. Cost is reported per sample.\n\n"
        + table
        + "\n"
    )


def replace_section(content: str, section_title_re: str, new_section: str) -> str:
    import re
    pattern = re.compile(
        rf'(\n### {section_title_re}\n.*?)(?=\n### |\n## |\Z)',
        re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(new_section, content)
    # Not found: insert before Takeaways
    if "\n## Takeaways\n" in content:
        return content.replace("\n## Takeaways\n", new_section + "\n## Takeaways\n", 1)
    return content.rstrip() + "\n" + new_section


def make_takeaway_subsection(title: str, section_desc: str, source_suffix: str) -> str:
    table_md = build_summary_table_for_source(source_suffix, section_desc, best_rag_note="")
    return (f"\n### {title}\n\n**All metrics, mean±std across 3 seeds "
            f"(noSF = no self-feedback, SF = self-feedback, Δ = SF − noSF):**\n{table_md}")


def replace_takeaways(content: str, new_takeaways: str) -> str:
    import re
    pattern = re.compile(r'\n## Takeaways\n.*', re.DOTALL)
    if pattern.search(content):
        return pattern.sub(new_takeaways, content)
    return content.rstrip() + "\n" + new_takeaways


def remove_section(content: str, heading: str) -> str:
    """Drop a whole '## <heading>' section (up to the next '## ' or EOF)."""
    import re
    pattern = re.compile(rf'\n## {re.escape(heading)}\n.*?(?=\n## |\Z)', re.DOTALL)
    return pattern.sub("", content)


def main() -> None:
    n_seeds_note = f"mean±std across {len(SEEDS)} seeds ({', '.join(str(s) for s in SEEDS)})"

    content = REPORT_PATH.read_text(encoding="utf-8")

    # Remove any previously appended section from old script runs
    old_marker = "\n## New mixed ES dev results (vLLM, seed-averaged)\n"
    if old_marker in content:
        start = content.index(old_marker)
        next_sec = content.find("\n## ", start + len(old_marker))
        content = content[:start] + (content[next_sec:] if next_sec != -1 else "")

    # The md now carries only the summary tables and the decision tables; the
    # per-experiment "main" tables under "## Current results" are dropped.
    content = remove_section(content, "Current results")

    # Build and replace Takeaways
    sns_tk = make_takeaway_subsection(
        "SNS1064 dev summary",
        f"SNS1064 dev set, 63 examples, open-answer task. vLLM, {n_seeds_note}.",
        "_sns1064",
    )
    casi_tk = make_takeaway_subsection(
        "CasiMedicos dev summary",
        f"CasiMedicos dev set, 63 examples, multiple-choice task. vLLM, {n_seeds_note}.",
        "_casimedicos",
    )
    mixed_tk = make_takeaway_subsection(
        "SNS1064+CasiMedicos dev summary",
        f"Mixed dev set, 126 examples (63 SNS1064 + 63 CasiMedicos). vLLM, {n_seeds_note}.",
        "",
    )
    new_takeaways = (
        "\n## Takeaways\n"
        + sns_tk
        + casi_tk
        + mixed_tk
        + build_meanq_decision_tables()
        + THINK_VS_NOTHINK_NOTE
    )
    content = replace_takeaways(content, new_takeaways)

    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Updated {REPORT_PATH}  (summary + decision tables; main tables removed)")


if __name__ == "__main__":
    main()
