#!/usr/bin/env python
"""
Replace the SNS1064+CasiMedicos EU section in eu_dev_ablation_results.md with
new rows from the vLLM runs (configs 1040-1050 Llama, 1051-1061 Latxa),
computing mean±std across seeds 42/43/44.

Table structure exactly matches the existing EU multi-model mixed table:
  #, model, experiment, self-feedback, [8 metrics × 3 sections], sec/sample, input, output, total
"""
from __future__ import annotations

import html
import json
import math
import re
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
)

METRICS_DIR = ROOT / "reports" / "metrics"
REPORT_PATH = ROOT / "reports" / "metrics" / "eu_dev_ablation_results.md"

SEEDS = [42, 43, 44]

EXPERIMENTS = [
    # (label, llama_id_prefix, llama_base, latxa_id_prefix, latxa_base)
    ("Baseline LLM only",              "1040", "llama31_8b_no_rag_extractive_mixed_eu_dev",              "1051", "latxa_llama31_8b_no_rag_extractive_mixed_eu_dev"),
    ("e5 top 1",                       "1041", "llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev",        "1052", "latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev"),
    ("e5 top 3",                       "1042", "llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev",        "1053", "latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev"),
    ("e5 top 5",                       "1043", "llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev",        "1054", "latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev"),
    ("rerank top 1",                   "1044", "llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev",      "1055", "latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev"),
    ("rerank top 3",                   "1045", "llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev",      "1056", "latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev"),
    ("rerank top 5",                   "1046", "llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev",      "1057", "latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev"),
    ("3-shot, no RAG",                 "1047", "llama31_8b_3shot_no_rag_extractive_mixed_eu_dev",        "1058", "latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev"),
    ("3-shot + rerank top 5",          "1048", "llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev","1059", "latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev"),
    ("cross-domain: SNS index",        "1049", "llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev","1060","latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev"),
    ("cross-domain: CasiMedicos index","1050", "llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev","1061","latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev"),
]

def build_meanq_decision_tables() -> str:
    """MeanQ-ranked decision table per model (Llama, Latxa), from meanq.py.

    Same source of truth as the staged ablation; regenerated, never hand-edited.
    Indices into EXPERIMENTS: 1 e5 top1 ... 6 rerank5 are the retrieval sweep the
    base RAG config is chosen from. EU tuple = (label, llama_id, llama_base,
    latxa_id, latxa_base).
    """
    from meanq import decision_table  # noqa: E402

    RETRIEVAL_IDX = [1, 2, 3, 4, 5, 6]
    labels = [EXPERIMENTS[i][0] for i in RETRIEVAL_IDX]

    def cand(pick_id, pick_base):
        return {lbl: (EXPERIMENTS[i][pick_id], EXPERIMENTS[i][pick_base])
                for i, lbl in zip(RETRIEVAL_IDX, labels)}

    parts = ["\n### Best RAG config by MeanQ (EU mixed dev task)\n",
             "MeanQ = mean(ROUGE-L, BERT-F1, MC-acc), averaged over 3 seeds on the "
             "no-self-feedback prediction. This is the score the staged ablation uses "
             "to choose the base RAG config that the few-shot and domain-restriction "
             "experiments are then wired to.\n"]
    parts.append(decision_table(cand(1, 2), title="Llama-3.1-8B-Instruct"))
    parts.append(decision_table(cand(3, 4), title="Latxa-Llama-3.1-8B-Instruct"))
    return "\n".join(parts)


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
        suffix (this is what the 1040-1061 EU runs used). That made "the new
        suffixless run" and "the old seed-42 run" the same path, so a re-run would
        overwrite the archive in place.

    The superseded runs live in experiments/runs_v2 / reports/metrics_v2.
    """
    return f"{id_prefix}_{base}_seed{seed}"


def load_summary(run: str, source_suffix: str = "") -> dict | None:
    p = METRICS_DIR / f"{run}{source_suffix}.json"
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
            # before_feedback holds initial (noSF) prediction metrics
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


def fmt_ms(m: float | None, s: float | None) -> str:
    if m is None:
        return ""
    return f"{m:.2f}±{s:.2f}" if s is not None else f"{m:.2f}"


def _sec_nosf(summary: dict | None) -> float | None:
    """Seconds per sample for the noSF (initial generation) phase only."""
    if not summary:
        return None
    t = summary.get("cost", {}).get("timing", {})
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


# ── table builder (mirrors write_eu_ablation_summary.build_table) ─────────────

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


def _best_per_model(all_rows: list[dict]) -> dict[str, str]:
    """Return {model_label: experiment_label} for the experiment with highest max(noSF, SF) BERTScore per model."""
    scores: dict[tuple[str, str], float] = {}
    for row in all_rows:
        val_str = row.get("overall_bertscore_f1", "")
        if not val_str:
            continue
        key = (row["model"], row.get("experiment", ""))
        mean_val = float(val_str.split("±")[0])
        scores[key] = max(scores.get(key, -1.0), mean_val)
    best: dict[str, tuple[float, str]] = {}
    for (model, exp), val in scores.items():
        if model not in best or val > best[model][0]:
            best[model] = (val, exp)
    return {model: exp for model, (_, exp) in best.items()}


def _best_overall(all_rows: list[dict]) -> tuple[str, str]:
    """Return (model_label, experiment_label) for the single best experiment overall by max(noSF, SF)."""
    scores: dict[tuple[str, str], float] = {}
    for row in all_rows:
        val_str = row.get("overall_bertscore_f1", "")
        if not val_str:
            continue
        key = (row["model"], row.get("experiment", ""))
        mean_val = float(val_str.split("±")[0])
        scores[key] = max(scores.get(key, -1.0), mean_val)
    if not scores:
        return "", ""
    (best_model, best_exp), _ = max(scores.items(), key=lambda kv: kv[1])
    return best_model, best_exp


def build_table(all_rows: list[dict], *, include_mc_accuracy: bool = False) -> str:
    best_per_model_map = _best_per_model(all_rows)
    best_overall_model, best_overall_exp = _best_overall(all_rows)

    quality_span = len(QUALITY_METRICS) * len(SECTIONS) + (1 if include_mc_accuracy else 0)
    header_1 = (
        th("#", rowspan=3)
        + th("model", rowspan=3)
        + th("experiment", rowspan=3)
        + th("self-feedback", rowspan=3)
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
        model = row["model"]
        exp = row.get("experiment", "")
        sf = row.get("sf", "")
        # Highlight both noSF and SF rows for the winning experiment.
        # Best determined by max(noSF, SF) overall BERTScore mean.
        is_best_overall = model == best_overall_model and exp == best_overall_exp
        is_best_model = best_per_model_map.get(model) == exp
        if is_best_overall:
            row_style = ' style="background-color: #d1f8d1;"'
        elif is_best_model:
            row_style = ' style="background-color: #fff8c5;"'
        else:
            row_style = ""
        cells = [
            td(row.get("label", "")),
            td(model),
            td(exp),
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

def build_rows_for_source(source_suffix: str) -> list[dict]:
    all_rows: list[dict] = []
    exp_num = 0
    for label, l_id, l_base, x_id, x_base in EXPERIMENTS:
        for model_label, id_prefix, base, model_suffix in [
            ("Llama-3.1-8B-Instruct", l_id, l_base, "a"),
            ("Latxa-Llama-3.1-8B-Instruct", x_id, x_base, "b"),
        ]:
            summaries = [load_summary(run_dir(id_prefix, base, s, source_suffix), source_suffix) for s in SEEDS]
            if not any(summaries):
                print(f"WARNING: no metrics for {id_prefix}_{base} suffix='{source_suffix}'")
                continue
            agg_nosf = aggregate_metrics(id_prefix, base, source_suffix=source_suffix, use_after=False)
            for sf_label, use_after in [("noSF", False), ("SF", True)]:
                row: dict[str, Any] = {}
                for sec in SECTIONS:
                    for met in QUALITY_METRICS:
                        vals = _vals(summaries, sec, met, use_after=use_after)
                        m_val, s_val = _mean_std(vals)
                        row[f"{sec}_{met}"] = fmt_ms(m_val, s_val)
                # mc_accuracy: CasiMedicos-only, short_answer section.
                mc_vals = _vals(summaries, "overall", "mc_accuracy", use_after=use_after)
                mc_m, mc_s = _mean_std(mc_vals)
                row["mc_accuracy"] = fmt_ms(mc_m, mc_s)
                # cost from same (merged SF/noSF) run
                row.update({k: agg_nosf.get(k, "") for k in
                             ("sec_per_sample", "mean_input_tokens",
                              "mean_output_tokens", "mean_total_tokens")})
                row.update({
                    "label": f"{exp_num}{model_suffix}" if sf_label == "noSF" else "",
                    "model": model_label,
                    "experiment": label,
                    "sf": sf_label,
                })
                all_rows.append(row)
        exp_num += 1
    return all_rows


def aggregate_metrics(id_prefix: str, base: str, source_suffix: str = "",
                      use_after: bool = False) -> dict[str, str]:
    summaries = [load_summary(run_dir(id_prefix, base, s, source_suffix), source_suffix) for s in SEEDS]
    available = [s for s in summaries if s]
    if not available:
        return {}
    row: dict[str, str] = {}
    for sec in SECTIONS:
        for met in QUALITY_METRICS:
            vals = _vals(summaries, sec, met, use_after=use_after)
            m, s = _mean_std(vals)
            row[f"{sec}_{met}"] = fmt_ms(m, s)

    def _sec(summary: dict | None) -> float | None:
        if not summary:
            return None
        t = summary.get("cost", {}).get("timing", {})
        v = t.get("example_seconds", {}).get("mean")
        if v is None:
            phases = ["retrieval_seconds", "rerank_seconds", "few_shot_seconds",
                      "prompt_seconds", "generation_seconds"]
            total = sum(float(t.get(p, {}).get("mean") or 0) for p in phases)
            v = total if total > 0 else None
        return float(v) if v is not None else None

    def _tok(summary: dict | None, key: str) -> float | None:
        if not summary:
            return None
        v = summary.get("cost", {}).get("token_counts", {}).get(key, {}).get("mean")
        return float(v) if v is not None else None

    sec_vals = [_sec(s) for s in summaries]
    m, s = _mean_std([v for v in sec_vals if v is not None])
    row["sec_per_sample"] = fmt_ms(m, s)
    in_vals = [_tok(s, "input_tokens") for s in summaries]
    out_vals = [_tok(s, "initial_output_tokens") for s in summaries]
    m, s = _mean_std([v for v in in_vals if v is not None])
    row["mean_input_tokens"] = fmt_ms(m, s)
    m, s = _mean_std([v for v in out_vals if v is not None])
    row["mean_output_tokens"] = fmt_ms(m, s)
    tot = [i + o for i, o in zip(in_vals, out_vals) if i is not None and o is not None]
    m, s = _mean_std(tot)
    row["mean_total_tokens"] = fmt_ms(m, s)
    return row


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


def build_summary_table_for_source(source_suffix: str, section_desc: str) -> str:
    """Build a compact markdown summary table (BERT noSF, BERT SF, SF Δ) for both Llama and Latxa.

    Best row per model is determined by highest BERT (SF) mean.
    """
    EXPERIMENT_LABELS = [lbl for lbl, *_ in EXPERIMENTS]
    models = [
        ("Llama-3.1-8B-Instruct",      [(l_id, l_base) for _, l_id, l_base, _, _ in EXPERIMENTS]),
        ("Latxa-Llama-3.1-8B-Instruct", [(x_id, x_base) for _, _, _, x_id, x_base in EXPERIMENTS]),
    ]
    include_mc_accuracy = source_suffix != "_sns1064"

    lines = [f"\n*{section_desc}*\n"]

    for model_label, id_bases in models:
        # Collect per-row values first so we can determine best in one pass
        row_data = []
        for id_prefix, base in id_bases:
            summaries = [load_summary(run_dir(id_prefix, base, s, source_suffix), source_suffix) for s in SEEDS]
            bm, bs = _mean_std(_vals(summaries, "overall", "bertscore_f1", use_after=False))
            am, as_ = _mean_std(_vals(summaries, "overall", "bertscore_f1", use_after=True))
            dm, ds = load_sf_delta(id_prefix, base, source_suffix)
            cost_nosf = _cost_nosf(summaries)
            cost_sf = _cost_sf(summaries)
            cost_delta = _cost_delta(summaries)
            mc_nosf_m, mc_nosf_s = _mean_std(_vals(summaries, "overall", "mc_accuracy", use_after=False))
            mc_sf_m, mc_sf_s = _mean_std(_vals(summaries, "overall", "mc_accuracy", use_after=True))
            row_data.append((bm, bs, am, as_, dm, ds, cost_nosf, cost_sf, cost_delta, mc_nosf_m, mc_nosf_s, mc_sf_m, mc_sf_s))

        # Best = highest max(noSF, SF) — i.e. the best achievable score for that config
        best_idx, best_val = -1, -1.0
        for i, (bm, _, am, *_rest) in enumerate(row_data):
            best_of_two = max(v for v in (bm, am) if v is not None) if any(v is not None for v in (bm, am)) else None
            if best_of_two is not None and best_of_two > best_val:
                best_val, best_idx = best_of_two, i

        lines.append(f"\n**{model_label}**\n")
        mc_header = "MC Accuracy (noSF) | MC Accuracy (SF) | " if include_mc_accuracy else ""
        mc_align = "---: | ---: | " if include_mc_accuracy else ""
        lines.append(f"| # | Experiment | {mc_header}BERT (noSF) | BERT (SF) | SF Δ | sec/sample (noSF) | sec/sample (SF) | sec/sample Δ | tokens/sample (noSF) | tokens/sample (SF) | tokens/sample Δ |")
        lines.append(f"|---:|---|{mc_align}---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, (bm, bs, am, as_, dm, ds, cost_nosf, cost_sf, cost_delta, mc_nosf_m, mc_nosf_s, mc_sf_m, mc_sf_s) in enumerate(row_data):
            bert_str = f"{bm:.2f}±{bs:.2f}" if bm is not None else ""
            sf_str = f"{am:.2f}±{as_:.2f}" if am is not None else ""
            delta_str = f"{dm:+.2f}±{ds:.2f}" if dm is not None else ""
            sec_nosf_str = cost_nosf.get("sec_per_sample", "")
            sec_sf_str = cost_sf.get("sec_per_sample", "")
            sec_delta_str = cost_delta.get("sec_per_sample", "")
            tok_nosf_str = cost_nosf.get("mean_total_tokens", "")
            tok_sf_str = cost_sf.get("mean_total_tokens", "")
            tok_delta_str = cost_delta.get("mean_total_tokens", "")
            lbl = EXPERIMENT_LABELS[i]
            mc_cells = ""
            if include_mc_accuracy:
                mc_nosf_str = f"{mc_nosf_m:.2f}±{mc_nosf_s:.2f}" if mc_nosf_m is not None else ""
                mc_sf_str = f"{mc_sf_m:.2f}±{mc_sf_s:.2f}" if mc_sf_m is not None else ""
                mc_cells = f"{mc_nosf_str} | {mc_sf_str} | "
            # Bold the winning column (noSF or SF, whichever is higher)
            if i == best_idx:
                nosf_is_best = bm is not None and (am is None or bm >= am)
                b_str = f"**{bert_str}**" if nosf_is_best else bert_str
                s_str = f"**{sf_str}**" if not nosf_is_best else sf_str
                lines.append(f"| **{i}** | **{lbl}** | {mc_cells}{b_str} | {s_str} | {delta_str} | "
                             f"{sec_nosf_str} | {sec_sf_str} | {sec_delta_str} | "
                             f"{tok_nosf_str} | {tok_sf_str} | {tok_delta_str} |")
            else:
                lines.append(f"| {i} | {lbl} | {mc_cells}{bert_str} | {sf_str} | {delta_str} | "
                             f"{sec_nosf_str} | {sec_sf_str} | {sec_delta_str} | "
                             f"{tok_nosf_str} | {tok_sf_str} | {tok_delta_str} |")

    return "\n".join(lines) + "\n"


def make_section(title: str, subtitle: str, source_suffix: str) -> str:
    rows = build_rows_for_source(source_suffix)
    if not rows:
        return ""
    include_mc_accuracy = source_suffix != "_sns1064"
    table = build_table(rows, include_mc_accuracy=include_mc_accuracy)
    return (
        f"\n### {title}\n\n"
        f"*{subtitle}*\n\n"
        "All quality scores are out of 100. Cost is reported per sample.\n\n"
        + table + "\n"
    )


def replace_section(content: str, title_re: str, new_section: str) -> str:
    pattern = re.compile(
        rf'(\n### {title_re}\n.*?)(?=\n### |\n## |\Z)',
        re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(new_section, content)
    if "\n## Takeaways\n" in content:
        return content.replace("\n## Takeaways\n", new_section + "\n## Takeaways\n", 1)
    return content.rstrip() + "\n" + new_section


def replace_or_append_takeaways(content: str, new_takeaways: str) -> str:
    pattern = re.compile(r'\n## Takeaways\n.*', re.DOTALL)
    if pattern.search(content):
        return pattern.sub(new_takeaways, content)
    return content.rstrip() + "\n" + new_takeaways


def main() -> None:
    n_seeds_note = f"mean±std across {len(SEEDS)} seeds ({', '.join(str(s) for s in SEEDS)})"

    sns_section = make_section(
        title="SNS1064 EU dev results",
        subtitle=(
            f"SNS1064 EU dev set, 63 examples, open-answer task (Basque). "
            f"vLLM inference, {n_seeds_note}. "
            "a = Llama-3.1-8B-Instruct, b = Latxa-Llama-3.1-8B-Instruct."
        ),
        source_suffix="_sns1064",
    )
    casi_section = make_section(
        title="CasiMedicos EU dev results",
        subtitle=(
            f"CasiMedicos EU dev set, 63 examples, multiple-choice task (Basque). "
            f"vLLM inference, {n_seeds_note}. "
            "a = Llama-3.1-8B-Instruct, b = Latxa-Llama-3.1-8B-Instruct."
        ),
        source_suffix="_casimedicos",
    )
    mixed_section = make_section(
        title="SNS1064+CasiMedicos EU dev results",
        subtitle=(
            f"SNS1064+CasiMedicos EU dev set, 126 examples (63 SNS1064-EU + 63 CasiMedicos-EU). "
            f"vLLM inference, {n_seeds_note}. "
            "a = Llama-3.1-8B-Instruct, b = Latxa-Llama-3.1-8B-Instruct."
        ),
        source_suffix="",
    )

    content = REPORT_PATH.read_text(encoding="utf-8")

    # Remove any previously appended section from old script runs
    old_marker = "\n## New mixed EU dev results (vLLM, seed-averaged)\n"
    if old_marker in content:
        start = content.index(old_marker)
        next_sec = content.find("\n## ", start + len(old_marker))
        content = content[:start] + (content[next_sec:] if next_sec != -1 else "")

    # Replace all three result sections
    content = replace_section(content, r"SNS1064 EU dev results", sns_section)
    content = replace_section(content, r"CasiMedicos EU dev results", casi_section)
    content = replace_section(content, r"SNS1064\+CasiMedicos(?: EU)? dev results", mixed_section)

    # Build and append/replace Takeaways
    sns_tk_body = build_summary_table_for_source(
        "_sns1064",
        f"SNS1064 EU dev set, 63 examples, open-answer task (Basque). vLLM, {n_seeds_note}.",
    )
    casi_tk_body = build_summary_table_for_source(
        "_casimedicos",
        f"CasiMedicos EU dev set, 63 examples, multiple-choice task (Basque). vLLM, {n_seeds_note}.",
    )
    mixed_tk_body = build_summary_table_for_source(
        "",
        f"Mixed EU dev set, 126 examples (63 SNS1064-EU + 63 CasiMedicos-EU). vLLM, {n_seeds_note}.",
    )
    new_takeaways = (
        "\n## Takeaways\n"
        f"\n### SNS1064 EU dev summary\n\n**Core findings (overall BERTScore, mean±std across 3 seeds; best row = highest max(noSF, SF)):**\n"
        + sns_tk_body
        + f"\n### CasiMedicos EU dev summary\n\n**Core findings (overall BERTScore, mean±std across 3 seeds; best row = highest max(noSF, SF)):**\n"
        + casi_tk_body
        + f"\n### SNS1064+CasiMedicos EU dev summary\n\n**Core findings (overall BERTScore, mean±std across 3 seeds; best row = highest max(noSF, SF)):**\n"
        + mixed_tk_body
        + build_meanq_decision_tables()
    )
    content = replace_or_append_takeaways(content, new_takeaways)

    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Updated {REPORT_PATH}  (3 result sections + Takeaways, {len(EXPERIMENTS)} experiments each)")


if __name__ == "__main__":
    main()
