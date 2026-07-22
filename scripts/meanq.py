#!/usr/bin/env python
"""MeanQ: the quality score used to choose the best RAG configuration.

MeanQ = mean(ROUGE-L, BERT-F1, MC-accuracy), each as its raw 0-100 value, averaged
over the three seeds, on the no-self-feedback (initial) prediction.

Rationale for the choice of metrics: ROUGE-L captures lexical fidelity, BERT-F1
captures semantic fidelity, and MC-accuracy captures decision correctness. Averaging
the three rewards a configuration only if it does well on all of the axes the thesis
cares about, rather than winning on one and losing on the others -- which a single
metric (e.g. BERTScore alone) can hide.

MC-accuracy is defined only for CasiMedicos-Exp (multiple-choice) records. On the
mixed dev set it is therefore read from the CasiMedicos subset (`_casimedicos.json`),
matching how the results tables report it. On an open-answer-only set (SNS-1064)
MC-accuracy does not exist and MeanQ is the mean of the two overlap metrics.

This module is pure computation over the metric JSONs; it loads no models and needs
no GPU. It is the single source of truth for "which RAG config is best" and is used
both to report the decision and to wire the dependent configs (few-shot, domain
restriction) to the correct base.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
SEEDS = [42, 43, 44]

QUALITY_METRICS = ("rouge_l_f1", "bertscore_f1", "mc_accuracy")


def _overall(summary: dict, use_sf: bool) -> dict:
    block = "after_feedback" if use_sf else "before_feedback"
    return (summary.get(block) or {}).get("overall") or summary.get("overall") or {}


def _load(run: str, suffix: str = "") -> Optional[dict]:
    p = METRICS / f"{run}{suffix}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("summary")


def metric_over_seeds(prefix: str, base: str, metric: str, *, suffix: str = "",
                      use_sf: bool = False) -> Optional[float]:
    vals = []
    for seed in SEEDS:
        summ = _load(f"{prefix}_{base}_seed{seed}", suffix)
        if not summ:
            continue
        v = _overall(summ, use_sf).get(metric)
        if v is not None:
            vals.append(float(v))
    return statistics.mean(vals) if vals else None


def metric_per_seed(prefix: str, base: str, metric: str, *, suffix: str = "",
                    use_sf: bool = False) -> list[Optional[float]]:
    """Per-seed values (None where a seed is missing the metric), seed order preserved.

    Unlike metric_over_seeds this keeps one entry per seed so callers can compute a
    std or pair the value with another metric from the same seed.
    """
    out: list[Optional[float]] = []
    for seed in SEEDS:
        summ = _load(f"{prefix}_{base}_seed{seed}", suffix)
        v = _overall(summ, use_sf).get(metric) if summ else None
        out.append(float(v) if v is not None else None)
    return out


def meanq_per_seed(prefix: str, base: str, *, use_sf: bool = False) -> list[Optional[float]]:
    """Per-seed MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) within each seed.

    ROUGE-L/BERT-F1 come from the mixed metric file; MC-acc from that same seed's
    CasiMedicos subset (undefined on the open-answer half). A seed's MeanQ is the
    mean of whichever of its three components exist; None if the seed has none.
    Keeping it per-seed lets the summary/decision tables report a real ±std on MeanQ.
    """
    rouge = metric_per_seed(prefix, base, "rouge_l_f1", use_sf=use_sf)
    bert = metric_per_seed(prefix, base, "bertscore_f1", use_sf=use_sf)
    mc = metric_per_seed(prefix, base, "mc_accuracy", suffix="_casimedicos", use_sf=use_sf)
    out: list[Optional[float]] = []
    for r, b, m in zip(rouge, bert, mc):
        present = [v for v in (r, b, m) if v is not None]
        out.append(statistics.mean(present) if present else None)
    return out


def meanq(prefix: str, base: str, *, use_sf: bool = False) -> tuple[Optional[float], dict]:
    """MeanQ on the MIXED dev set. Returns (MeanQ, per-metric components).

    ROUGE-L and BERT-F1 come from the mixed metric file; MC-accuracy from the
    CasiMedicos subset (it is undefined on the open-answer half). MeanQ is the mean
    of whichever of the three are available.
    """
    rouge = metric_over_seeds(prefix, base, "rouge_l_f1", use_sf=use_sf)
    bert = metric_over_seeds(prefix, base, "bertscore_f1", use_sf=use_sf)
    mc = metric_over_seeds(prefix, base, "mc_accuracy", suffix="_casimedicos", use_sf=use_sf)
    components = {"rouge_l_f1": rouge, "bertscore_f1": bert, "mc_accuracy": mc}
    present = [v for v in components.values() if v is not None]
    return (statistics.mean(present) if present else None), components


def best_by_meanq(candidates: dict[str, tuple[str, str]], *, use_sf: bool = False
                  ) -> tuple[Optional[str], dict[str, float]]:
    """Given {label: (id_prefix, base)}, return (winning label, {label: MeanQ}).

    A configuration with no metrics yet is skipped, so this is safe to call before
    every run has been evaluated; it just picks the best of what exists.
    """
    scores: dict[str, float] = {}
    for label, (prefix, base) in candidates.items():
        mq, _ = meanq(prefix, base, use_sf=use_sf)
        if mq is not None:
            scores[label] = mq
    if not scores:
        return None, {}
    winner = max(scores, key=scores.get)
    return winner, scores


def relative_cost(prefix: str, base: str) -> float:
    """A monotone proxy for a RAG config's inference cost: the number of
    documents actually retrieved/embedded into the prompt (retrieval_top_k),
    plus a fixed penalty for running the cross-encoder reranker (it scores
    every one of the retrieval_top_k candidates -- typically 15 in this grid
    -- regardless of how many survive to reranker_top_k, so that full pass is
    real wall-clock cost the final top_k number alone would hide).

    Not calibrated to actual latency/FLOPs; only used to break near-ties in
    MeanQ toward the cheaper of two options that score almost the same.
    """
    config_path = ROOT / "configs" / "experiments" / f"{prefix}_{base}.json"
    if not config_path.exists():
        return float("inf")
    config = json.loads(config_path.read_text())
    cost = float(config.get("retrieval_top_k") or 0)
    if config.get("reranker_model"):
        cost += 5.0  # fixed reranker-pass penalty, on the same rough scale as top_k
    return cost


def best_by_meanq_robust(
    candidates: dict[str, tuple[str, str]], *, use_sf: bool = False,
    margin: float = 0.5, std_ratio: float = 0.4, cost_ratio: float = 0.4,
) -> tuple[Optional[str], dict[str, dict]]:
    """Variance- and cost-aware version of best_by_meanq, using a pairwise,
    point-scored tie-break (see manuscript \\S "Selecting the best
    configuration" / sec:selection-rule for the formal statement).

    Reduces to the plain highest-mean-MeanQ winner whenever the leader's mean
    is at least `margin` points clear of every other candidate. Otherwise, the
    leader is compared pairwise against each candidate within `margin` points:
    each of standard deviation and cost independently awards one point to
    whichever side wins it by more than `std_ratio` / `cost_ratio` (as a
    fraction of the larger of the two values on that axis) -- a criterion that
    doesn't clear its own threshold awards no point to either side. Whichever
    side has more points after both criteria is preferred; if the two split
    one point each, or neither criterion is decisive, the pairwise winner
    falls back to whichever of the two has the higher mean MeanQ (even though,
    by construction, that difference is itself under `margin`) -- a real, if
    marginal, quality edge is never discarded once stability/cost are
    themselves inconclusive. The leader is replaced by the pairwise winner and
    the process repeats against the next candidate, so the final winner has
    beaten every other candidate under this rule.

    Returns (winning label, {label: {"mean": ..., "std": ..., "cost": ..., "n": ...}}).
    """
    stats: dict[str, dict] = {}
    for label, (prefix, base) in candidates.items():
        per_seed = meanq_per_seed(prefix, base, use_sf=use_sf)
        values = [v for v in per_seed if v is not None]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[label] = {
            "mean": mean, "std": std, "n": len(values),
            "cost": relative_cost(prefix, base),
        }
    if not stats:
        return None, {}

    def pairwise_winner(a: str, b: str) -> str:
        """Which of a, b wins under the point-scored rule, a vs b."""
        mean_a, mean_b = stats[a]["mean"], stats[b]["mean"]
        if abs(mean_a - mean_b) >= margin:
            return a if mean_a > mean_b else b

        std_a, std_b = stats[a]["std"], stats[b]["std"]
        cost_a, cost_b = stats[a]["cost"], stats[b]["cost"]
        points = {a: 0, b: 0}

        std_max = max(std_a, std_b)
        if std_max > 0:
            if std_b - std_a > std_ratio * std_max:
                points[a] += 1
            elif std_a - std_b > std_ratio * std_max:
                points[b] += 1

        cost_max = max(cost_a, cost_b)
        if cost_max > 0:
            if cost_b - cost_a > cost_ratio * cost_max:
                points[a] += 1
            elif cost_a - cost_b > cost_ratio * cost_max:
                points[b] += 1

        if points[a] != points[b]:
            return a if points[a] > points[b] else b
        return a if mean_a >= mean_b else b

    labels = list(stats)
    winner = labels[0]
    for label in labels[1:]:
        winner = pairwise_winner(winner, label)
    return winner, stats


def decision_table(candidates: dict[str, tuple[str, str]], *, title: str,
                   use_sf: bool = False, top: Optional[int] = None) -> str:
    """Render a MeanQ-ranked markdown decision table for one model+language.

    `candidates` maps a display label -> (id_prefix, base). Rows are sorted by
    MeanQ descending; the top row is bolded as the chosen configuration. This is
    the single source of the decision prose, so the .md reports never drift from
    what the staged ablation actually selected (see best_by_meanq).
    """
    rows = []
    for label, (prefix, base) in candidates.items():
        mq, comp = meanq(prefix, base, use_sf=use_sf)
        if mq is None:
            continue
        rows.append((label, mq, comp))
    rows.sort(key=lambda r: -r[1])
    if top is not None:
        rows = rows[:top]

    def cell(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "--"

    out = [f"\n### {title}\n",
           "| Rank | Experiment | ROUGE-L | BERT-F1 | MC-acc | **MeanQ** |",
           "|---:|---|---:|---:|---:|---:|"]
    for i, (label, mq, comp) in enumerate(rows):
        r, b, m = comp["rouge_l_f1"], comp["bertscore_f1"], comp["mc_accuracy"]
        line = (f"| {i+1} | {label} | {cell(r)} | {cell(b)} | {cell(m)} | "
                f"{cell(mq)} |")
        if i == 0:
            line = (f"| **{i+1}** | **{label}** | {cell(r)} | {cell(b)} | "
                    f"{cell(m)} | **{cell(mq)}** |")
        out.append(line)
    if rows:
        best_label, best_mq, _ = rows[0]
        out.append(
            f"\n**Chosen config: {best_label}** (MeanQ {best_mq:.2f}), the highest "
            "MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) over 3 seeds on the no-self-feedback "
            "prediction. MeanQ is used instead of any single metric so a config is only "
            "chosen if it does well on lexical, semantic, and decision correctness "
            "together.\n")
    return "\n".join(out)


if __name__ == "__main__":
    # Smoke: print MeanQ for the Llama retrieval configs.
    cand = {
        "retrieve top1": ("1041", "llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev"),
        "retrieve top3": ("1042", "llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev"),
        "retrieve top5": ("1043", "llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev"),
        "rerank1": ("1044", "llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev"),
        "rerank3": ("1045", "llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev"),
        "rerank5": ("1046", "llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev"),
    }
    winner, scores = best_by_meanq(cand)
    for label, mq in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {label:10} MeanQ={mq:.2f}")
    print(f"  best: {winner}")
