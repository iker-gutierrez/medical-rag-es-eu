#!/usr/bin/env python
"""Summarise the reasoning-pipeline dev runs against the frozen best RAG rows.

Both tables hold retrieval fixed (e5 top-15 -> rerank top-5) and vary only the
reasoning pipeline, so every delta here is attributable to the pipeline. Row 0 of
each table is the single-pass RAG baseline that won the corresponding dev
decision table, reproduced from its own metric files -- not re-run.

Baselines carry a self-feedback stage, so they have both a noSF and an SF column.
The reasoning pipelines have neither: the pipeline *is* the refinement mechanism.
They are therefore compared against the baseline's better column, and the table
says which one that was.

Writes reports/metrics/reasoning_pipeline_dev_results.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

METRICS_DIR = ROOT / "reports" / "metrics"
RUNS_DIR = ROOT / "experiments" / "runs"
REPORT_PATH = METRICS_DIR / "reasoning_pipeline_dev_results.md"

SEEDS = [42, 43, 44]

# (label, run-dir stem). The baseline is the winning row of each decision table.
# ES switched from Qwen3.5-9B think+rerank5 to no-think+rerank5 (MeanQ within noise
# of think's own row-8 winner, ~1/3 the cost); EU switched from Llama+e5-top3 to
# Latxa+e5-top1 (Latxa's actual ablation winner, corrected after the MC-acc fix).
ES_ROWS = [
    ("Baseline: single-pass RAG (no-think, rerank top 5)", "1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev", True),
    ("Structured clinical CoT (MedCoT-RAG)", "1330_qwen35_9b_structured_cot_e5_rerank5_no_think_extractive_mixed_dev", False),
    ("Thought-driven retrieval (RAR2, 1 round)", "1331_qwen35_9b_thought_rag_e5_rerank5_no_think_extractive_mixed_dev", False),
    ("Thought-driven retrieval (RAR2, iterative)", "1332_qwen35_9b_thought_rag_iter_e5_rerank5_no_think_extractive_mixed_dev", False),
    ("Multi-round agentic RAG (MA-RAG)", "1333_qwen35_9b_marag_e5_rerank5_no_think_extractive_mixed_dev", False),
]

EU_ROWS = [
    ("Baseline: single-pass RAG (Latxa, e5 top 1)", "1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev", True),
    ("Structured clinical CoT (MedCoT-RAG)", "1320_latxa_llama31_8b_structured_cot_e5_topk1_extractive_mixed_eu_dev", False),
    ("Thought-driven retrieval (RAR2, 1 round)", "1321_latxa_llama31_8b_thought_rag_e5_topk1_extractive_mixed_eu_dev", False),
    ("Thought-driven retrieval (RAR2, iterative)", "1322_latxa_llama31_8b_thought_rag_iter_e5_topk1_extractive_mixed_eu_dev", False),
    ("Multi-round agentic RAG (MA-RAG)", "1323_latxa_llama31_8b_marag_e5_topk1_extractive_mixed_eu_dev", False),
]

TASKS = [
    ("", "Mixed dev set (126 = 63 SNS1064 + 63 CasiMedicos)"),
    ("_sns1064", "SNS1064 only (63, open answer)"),
    ("_casimedicos", "CasiMedicos only (63, multiple choice)"),
]


def seed_runs(stem: str) -> list[str]:
    """Runs are written to _seed{N} dirs; seed 42 may also exist under the bare
    name (the convention the older single-pass runs used)."""
    runs = []
    for seed in SEEDS:
        explicit = f"{stem}_seed{seed}"
        if (METRICS_DIR / f"{explicit}.json").exists():
            runs.append(explicit)
        elif seed == 42 and (METRICS_DIR / f"{stem}.json").exists():
            runs.append(stem)
    return runs


def load_summary(run: str, suffix: str) -> Optional[dict[str, Any]]:
    path = METRICS_DIR / f"{run}{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("summary")


def metric_values(
    summaries: list[dict[str, Any]], section: str, metric: str, *, use_after: bool
) -> list[float]:
    values = []
    for summary in summaries:
        if use_after:
            value = (summary.get("after_feedback") or {}).get(section, {}).get(metric)
        else:
            value = (summary.get("before_feedback") or {}).get(section, {}).get(metric)
            if value is None:
                value = (summary.get(section) or {}).get(metric)
        if value is not None:
            values.append(float(value))
    return values


def mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def fmt(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "--"
    return f"{mean:.2f}±{std:.2f}" if std is not None else f"{mean:.2f}"


def cost_value(summaries: list[dict[str, Any]], field: str, *, token: bool) -> Optional[float]:
    values = []
    for summary in summaries:
        cost = summary.get("cost") or {}
        block = (cost.get("token_counts") if token else cost.get("timing")) or {}
        value = (block.get(field) or {}).get("mean")
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def collect(stem: str, suffix: str, *, use_after: bool) -> Optional[dict[str, Any]]:
    """One row. `use_after=True` reads the baseline's post-self-feedback numbers.

    The self-feedback baseline is reported as TWO rows (noSF and SF) rather than
    collapsed to whichever looks better: the reasoning pipelines have no
    self-feedback stage, so noSF is the like-for-like compute comparison while SF
    is the literal winning row of the dev decision table. Showing only one of them
    would quietly flatter or punish the pipelines depending on which was chosen.

    Metrics match the ablation result tables exactly: quality = ROUGE-L, BERT-F1,
    MC-acc, MeanQ; cost = sec/sample, tok/sample. Nothing else (no BERTScore-delta,
    Cosim, or token-F1) is a core column here, for the same reason the ablation
    tables dropped them -- see scripts/meanq.py and scripts/metric_tables.py.
    """
    runs = seed_runs(stem)
    summaries = [s for s in (load_summary(run, suffix) for run in runs) if s]
    if not summaries:
        return None

    row: dict[str, Any] = {"n_seeds": len(summaries), "column": "SF" if use_after else "noSF"}
    for metric in ("bertscore_f1", "rouge_l_f1"):
        mean, std = mean_std(metric_values(summaries, "overall", metric, use_after=use_after))
        row[metric] = (mean, std)

    # MC-acc: on the mixed set it only exists on the CasiMedicos half, read from
    # the _casimedicos metric subset -- matching meanq.py, so MeanQ here is
    # computed the same way as in the ablation decision tables. On a suffix that
    # IS already _casimedicos, use it directly; on _sns1064 there is no MC-acc.
    mc_suffix = "_casimedicos" if suffix == "" else (suffix if suffix == "_casimedicos" else None)

    def value_for(run: str, load_suffix: str, section: str, metric: str) -> Optional[float]:
        summary = load_summary(run, load_suffix)
        if summary is None:
            return None
        block = (summary.get("after_feedback") if use_after else summary.get("before_feedback")) or {}
        value = block.get(section, {}).get(metric)
        if value is None and not use_after:
            value = (summary.get(section) or {}).get(metric)
        return float(value) if value is not None else None

    mc_values = [value_for(run, mc_suffix, "overall", "mc_accuracy") for run in runs] if mc_suffix else [None] * len(runs)
    row["mc_accuracy"] = mean_std([v for v in mc_values if v is not None])

    # MeanQ per seed -- pairs ROUGE-L/BERT-F1/MC-acc from the SAME run/seed by
    # indexing on `runs` directly (not on metric_values()'s filtered output,
    # which drops Nones and would silently misalign seeds if one run is missing
    # a metric another has) -- then averages across seeds. Same method as
    # metric_tables.py's meanq_per_seed, so directly comparable to the ablation
    # decision tables.
    per_seed_meanq = []
    for i, run in enumerate(runs):
        rouge = value_for(run, suffix, "overall", "rouge_l_f1")
        bert = value_for(run, suffix, "overall", "bertscore_f1")
        mc = mc_values[i]
        parts = [v for v in (rouge, bert, mc) if v is not None]
        if parts:
            per_seed_meanq.append(sum(parts) / len(parts))
    row["meanq"] = mean_std(per_seed_meanq)

    row["sec_per_sample"] = cost_value(summaries, "example_seconds", token=False)
    row["total_tokens"] = cost_value(summaries, "total_tokens", token=True)

    # Not a core metric column (kept out of the ablation-matching set), but still
    # useful context for the pipelines specifically: how many LLM calls did each
    # answer actually cost, and (MA-RAG only) how often did candidates just agree.
    calls = []
    consensus = []
    for run in runs:
        meta_path = RUNS_DIR / run / "predictions.meta.json"
        if not meta_path.exists():
            continue
        run_meta = json.loads(meta_path.read_text())
        if run_meta.get("mean_llm_calls_per_record") is not None:
            calls.append(float(run_meta["mean_llm_calls_per_record"]))
        stats = run_meta.get("marag_stats") or {}
        total = run_meta.get("num_records") or 0
        if stats.get("num_reached_consensus") is not None and total:
            consensus.append(100.0 * float(stats["num_reached_consensus"]) / total)
    row["llm_calls"] = sum(calls) / len(calls) if calls else None
    row["consensus_pct"] = sum(consensus) / len(consensus) if consensus else None
    return row


def build_table(rows: list[tuple[str, str, bool]], suffix: str, include_mc: bool) -> str:
    """Same six metrics as the ablation decision tables: quality = ROUGE-L,
    BERT-F1, MC-acc, MeanQ; cost = sec/sample, tok/sample. The baseline
    (rows[0], has_sf=True) expands into two rows (noSF, SF); pipelines have no
    self-feedback stage of their own, so they get one row each, shown against
    both baseline rows via a MeanQ Δ (the composite score, not just BERTScore)."""
    collected: list[tuple[str, Optional[dict[str, Any]]]] = []
    for label, stem, has_sf in rows:
        if has_sf:
            collected.append((f"{label} — noSF", collect(stem, suffix, use_after=False)))
            collected.append((f"{label} — SF", collect(stem, suffix, use_after=True)))
        else:
            collected.append((label, collect(stem, suffix, use_after=False)))

    num_baselines = sum(1 for _, _, has_sf in rows if has_sf) * 2
    base_nosf = collected[0][1]["meanq"][0] if collected[0][1] else None
    base_sf = collected[1][1]["meanq"][0] if num_baselines > 1 and collected[1][1] else None

    mc_cols = ("MC-acc |", "---:|") if include_mc else ("", "")
    header = (
        "| # | Pipeline | col | ROUGE-L | BERT-F1 | " + mc_cols[0] + " MeanQ | "
        "Δ MeanQ vs noSF | Δ MeanQ vs SF | sec/sample | tok/sample | seeds |"
    )
    sep = "|---:|---|---|---:|---:|" + mc_cols[1] + "---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]

    n_cols = 8 + (1 if include_mc else 0)
    for idx, (label, row) in enumerate(collected):
        is_baseline = idx < num_baselines
        if row is None:
            lines.append(
                "| " + " | ".join([str(idx), label, "--", "*not yet run*"] + ["--"] * (n_cols - 1)) + " |"
            )
            continue
        meanq = row["meanq"]

        def delta(base: Optional[float]) -> str:
            if is_baseline or base is None or meanq[0] is None:
                return "baseline" if is_baseline else "--"
            return f"{meanq[0] - base:+.2f}"

        cells = [
            str(idx),
            f"**{label}**" if is_baseline else label,
            row["column"],
            fmt(*row["rouge_l_f1"]),
            fmt(*row["bertscore_f1"]),
        ]
        if include_mc:
            cells.append(fmt(*row["mc_accuracy"]))
        cells += [
            fmt(*meanq),
            delta(base_nosf),
            delta(base_sf),
            f"{row['sec_per_sample']:.2f}" if row["sec_per_sample"] is not None else "--",
            f"{row['total_tokens']:.0f}" if row["total_tokens"] is not None else "--",
            str(row["n_seeds"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # Not core metric columns, but useful pipeline-specific context.
    marag = next((row for label, row in collected if row and row.get("consensus_pct") is not None), None)
    calls_rows = [(label, row["llm_calls"]) for label, row in collected if row and row.get("llm_calls") is not None]
    if calls_rows:
        lines.append("")
        lines.append("*LLM calls per answer: " + "; ".join(f"{label} {v:.1f}" for label, v in calls_rows) + ".*")
    if marag is not None:
        lines.append(
            f"*MA-RAG reached candidate consensus (no synthesis needed) on "
            f"{marag['consensus_pct']:.1f}% of records.*"
        )
    return "\n".join(lines)


def main() -> None:
    parts = [
        "# Reasoning pipelines on the frozen best RAG configuration",
        "",
        "Retrieval is held fixed at the winning row of each dev decision table -- a "
        "*different* row per language (see below) -- so every difference here is "
        "attributable to the reasoning pipeline, not to retrieval.",
        "",
        "- **ES**: Qwen3.5-9B (no-think), retrieval e5 top-15 -> rerank top-5, Spanish.",
        "- **EU**: Latxa-Llama-3.1-8B-Instruct, retrieval **e5 top-1, no reranker**, Basque.",
        "",
        "Each language's retrieval is the winning row of its own decision table, ranked by "
        "MeanQ = mean(ROUGE-L, BERT-F1, MC-accuracy). ES uses Qwen3.5-9B no-think rather than "
        "think mode: think mode's MeanQ edge over no-think+rerank5 was 0.01 (well inside the "
        "seed-to-seed std), at roughly 3x the tokens and latency per call, and MA-RAG's own "
        "conflict-resolution mechanism is designed to close the same kind of gap think mode "
        "buys through multi-candidate sampling, so the cost was judged not worth it. EU uses "
        "Latxa (the Basque-adapted model) at e5 top-1, its own ablation winner -- fewer "
        "retrieved passages consistently raised MC-accuracy for Basque even though it lowered "
        "ROUGE-L/BERT-F1, the opposite of the pattern in Spanish.",
        "",
        "The baseline runs a self-feedback pass, so it appears as **two rows**: `noSF` (a "
        "single generation -- the like-for-like compute comparison, since the reasoning "
        "pipelines have no self-feedback stage) and `SF` (the literal winning row of the "
        "decision table, which gets a second refinement pass). Every pipeline is scored "
        "against both. `LLM calls` is the mean number of generations spent per answer -- the "
        "honest cost of a multi-round pipeline against a single-pass baseline.",
        "",
        "Pipelines: **MedCoT-RAG** (wangEtAl2025) structured clinical CoT; **RAR2** "
        "(xuEtAl2025) thought-driven retrieval, tuning-free `w/o training` ablation, in "
        "single-retrieval and iterative-scaling variants; **MA-RAG** (wuEtAl2026) "
        "conflict-guided multi-round agentic RAG, adapted to generative QA "
        "(semantic conflict for open answers, option disagreement for multiple choice).",
    ]

    for language, rows in (("Spanish (ES)", ES_ROWS), ("Basque (EU)", EU_ROWS)):
        parts += ["", f"## {language}"]
        for suffix, title in TASKS:
            include_mc = suffix != "_sns1064"
            parts += ["", f"### {title}", "", build_table(rows, suffix, include_mc)]

    REPORT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
