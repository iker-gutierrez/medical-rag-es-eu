#!/usr/bin/env python
"""Shared renderer for the .md summary and decision tables.

Both languages (ES: Mistral / Qwen no-think / Qwen think; EU: Llama / Latxa) and
both table kinds (per-model *summary* over all experiments, and the MeanQ-ranked
*decision* table over the retrieval sweep) show the same six metrics, each split
three ways -- no-self-feedback (noSF), self-feedback (SF), and their delta:

    quality: ROUGE-L, BERT-F1, MC-acc, MeanQ      (0-100, higher better)
    cost:    sec/sample, tok/sample               (per sample, lower better)

MeanQ is mean(ROUGE-L, BERT-F1, MC-acc) computed per seed (see meanq.py) so it
carries a real seed-to-seed std like the other columns. MC-acc and hence MeanQ
are undefined on the open-answer-only (SNS1064) split; those cells render blank.

This module owns only the table markup and the seed aggregation of quality
metrics. Cost aggregation differs slightly between the two callers (they already
had matching _cost_nosf/_cost_sf/_cost_delta helpers), so the caller passes those
three functions in rather than this module reimplementing them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from meanq import metric_per_seed, meanq_per_seed  # noqa: E402

# Column order and headers. Each maps to a per-seed value list producer below.
QUALITY = [
    ("rouge_l_f1", "ROUGE-L"),
    ("bertscore_f1", "BERT-F1"),
    ("mc_accuracy", "MC-acc"),
    ("meanq", "MeanQ"),
]


def _mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, s


def _fmt(m: Optional[float], s: Optional[float], *, signed: bool = False) -> str:
    if m is None:
        return ""
    core = f"{m:+.2f}" if signed else f"{m:.2f}"
    return f"{core}±{s:.2f}" if s is not None else core


def _quality_triplet(prefix: str, base: str, metric: str) -> tuple[str, str, str, Optional[float]]:
    """(noSF, SF, Δ) formatted strings for one quality metric, plus the noSF mean
    (used for MeanQ-ranking). MeanQ uses the per-seed composite; the rest read the
    metric directly. MC-acc comes from the CasiMedicos subset so it lines up with
    MeanQ's MC-acc component on the mixed table."""
    if metric == "meanq":
        nosf = meanq_per_seed(prefix, base, use_sf=False)
        sf = meanq_per_seed(prefix, base, use_sf=True)
    else:
        suffix = "_casimedicos" if metric == "mc_accuracy" else ""
        nosf = metric_per_seed(prefix, base, metric, suffix=suffix, use_sf=False)
        sf = metric_per_seed(prefix, base, metric, suffix=suffix, use_sf=True)
    nm, ns = _mean_std(nosf)
    sm, ss = _mean_std(sf)
    deltas = [a - b for a, b in zip(sf, nosf) if a is not None and b is not None]
    dm, ds = _mean_std(deltas)
    return _fmt(nm, ns), _fmt(sm, ss), _fmt(dm, ds, signed=True), nm


HEADER_CELLS = (
    [f"{h} (noSF)" for _, h in QUALITY]  # noqa: F841 -- built inline below instead
)


def _build_header() -> tuple[str, str]:
    cols = ["#", "Experiment"]
    for _, h in QUALITY:
        cols += [f"{h} (noSF)", f"{h} (SF)", f"{h} Δ"]
    cols += ["sec/sample (noSF)", "sec/sample (SF)", "sec/sample Δ",
             "tok/sample (noSF)", "tok/sample (SF)", "tok/sample Δ"]
    align = ["---:", "---"] + ["---:"] * (len(cols) - 2)
    return "| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"


def render_model_table(
    rows: list[tuple[int, str, str, str]],
    *,
    cost_nosf: Callable,
    cost_sf: Callable,
    cost_delta: Callable,
    load_summaries: Callable[[str, str], list],
    include_mc: bool,
    rank_by_meanq: bool = False,
) -> str:
    """Render one model's table.

    rows: list of (display_number, label, id_prefix, base). For the decision table
    display_number is the rank (filled here); for the summary it is the experiment #.

    cost_*: the caller's cost aggregators, each taking the seed summaries list and
    returning a dict with "sec_per_sample" and "mean_total_tokens".
    load_summaries(id_prefix, base) -> the per-seed summary dicts for cost.
    include_mc: blank the MC-acc and MeanQ columns on SNS1064-only tables.
    rank_by_meanq: sort rows by noSF MeanQ descending and renumber as rank; bold row 1.
    """
    built = []
    for number, label, prefix, base in rows:
        quality: dict[str, tuple[str, str, str]] = {}
        meanq_nosf = None
        for metric, _ in QUALITY:
            n, s, d, nm = _quality_triplet(prefix, base, metric)
            if not include_mc and metric in ("mc_accuracy", "meanq"):
                n = s = d = ""
                nm = None
            quality[metric] = (n, s, d)
            if metric == "meanq":
                meanq_nosf = nm
        summaries = load_summaries(prefix, base)
        cn, cs, cd = cost_nosf(summaries), cost_sf(summaries), cost_delta(summaries)
        built.append((number, label, quality, cn, cs, cd, meanq_nosf))

    if rank_by_meanq:
        built.sort(key=lambda r: (r[6] is None, -(r[6] or 0.0)))

    header, align = _build_header()
    lines = [header, align]
    for rank, (number, label, quality, cn, cs, cd, _mq) in enumerate(built, start=1):
        num = rank if rank_by_meanq else number
        best = rank_by_meanq and rank == 1
        cells: list[str] = []
        for metric, _ in QUALITY:
            cells += list(quality[metric])
        cells += [cn.get("sec_per_sample", ""), cs.get("sec_per_sample", ""),
                  cd.get("sec_per_sample", ""),
                  cn.get("mean_total_tokens", ""), cs.get("mean_total_tokens", ""),
                  cd.get("mean_total_tokens", "")]
        num_cell = f"**{num}**" if best else str(num)
        lbl_cell = f"**{label}**" if best else label
        lines.append("| " + " | ".join([num_cell, lbl_cell] + cells) + " |")
    return "\n".join(lines)
