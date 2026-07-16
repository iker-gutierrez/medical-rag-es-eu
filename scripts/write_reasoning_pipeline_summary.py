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
    """
    runs = seed_runs(stem)
    summaries = [s for s in (load_summary(run, suffix) for run in runs) if s]
    if not summaries:
        return None

    row: dict[str, Any] = {"n_seeds": len(summaries), "column": "SF" if use_after else "noSF"}
    for metric in ("bertscore_f1", "cosine_similarity", "rouge_l_f1", "token_overlap_f1"):
        for section in ("overall", "short_answer", "evidence"):
            mean, std = mean_std(metric_values(summaries, section, metric, use_after=use_after))
            row[f"{section}_{metric}"] = (mean, std)
    mc_mean, mc_std = mean_std(metric_values(summaries, "overall", "mc_accuracy", use_after=use_after))
    row["mc_accuracy"] = (mc_mean, mc_std)

    row["sec_per_sample"] = cost_value(summaries, "example_seconds", token=False)
    row["total_tokens"] = cost_value(summaries, "total_tokens", token=True)

    # Pipeline-only: how many LLM calls did each answer actually cost? The metric
    # JSON does not carry run metadata, so read it from the run's own meta file.
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
    # The baseline (rows[0], has_sf=True) expands into two rows: noSF and SF.
    collected: list[tuple[str, Optional[dict[str, Any]]]] = []
    for label, stem, has_sf in rows:
        if has_sf:
            collected.append((f"{label} — noSF", collect(stem, suffix, use_after=False)))
            collected.append((f"{label} — SF", collect(stem, suffix, use_after=True)))
        else:
            collected.append((label, collect(stem, suffix, use_after=False)))

    num_baselines = sum(1 for _, _, has_sf in rows if has_sf) * 2
    base_nosf = collected[0][1]["overall_bertscore_f1"][0] if collected[0][1] else None
    base_sf = collected[1][1]["overall_bertscore_f1"][0] if num_baselines > 1 and collected[1][1] else None

    header = (
        "| # | Pipeline | col | BERTScore | Δ vs noSF | Δ vs SF | Cosim | ROUGE-L | token F1 |"
        + (" MC Acc. |" if include_mc else "")
        + " sec/sample | tokens/sample | LLM calls | seeds |"
    )
    sep = (
        "|---:|---|---|---:|---:|---:|---:|---:|---:|"
        + ("---:|" if include_mc else "")
        + "---:|---:|---:|---:|"
    )
    lines = [header, sep]

    for idx, (label, row) in enumerate(collected):
        is_baseline = idx < num_baselines
        if row is None:
            lines.append(
                "| " + " | ".join([str(idx), label, "--", "*not yet run*"]
                                  + ["--"] * (9 + (1 if include_mc else 0))) + " |"
            )
            continue
        bert = row["overall_bertscore_f1"]

        def delta(base: Optional[float]) -> str:
            if is_baseline or base is None or bert[0] is None:
                return "baseline" if is_baseline else "--"
            return f"{bert[0] - base:+.2f}"

        cells = [
            str(idx),
            f"**{label}**" if is_baseline else label,
            row["column"],
            fmt(*bert),
            delta(base_nosf),
            delta(base_sf),
            fmt(*row["overall_cosine_similarity"]),
            fmt(*row["overall_rouge_l_f1"]),
            fmt(*row["overall_token_overlap_f1"]),
        ]
        if include_mc:
            cells.append(fmt(*row["mc_accuracy"]))
        cells += [
            f"{row['sec_per_sample']:.2f}" if row["sec_per_sample"] is not None else "--",
            f"{row['total_tokens']:.0f}" if row["total_tokens"] is not None else "--",
            f"{row['llm_calls']:.1f}" if row["llm_calls"] is not None else "1.0",
            str(row["n_seeds"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # MA-RAG's headline diagnostic: how often the candidates simply agreed.
    marag = next((row for label, row in collected if row and row.get("consensus_pct") is not None), None)
    if marag is not None:
        lines.append("")
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
