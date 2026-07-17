#!/usr/bin/env python
"""Generate the LaTeX result tables: four staged main-text tables + full appendix tables.

Two decisions are baked in here.

**MC accuracy replaces cosine similarity.** Cosine similarity over pooled sentence
embeddings is the least discriminative of the three quality metrics: it rewards a
fluent, on-topic, factually wrong answer almost as much as a correct one, and in
practice it saturates (most configurations land within a point or two of each
other, so the column carries little signal). Multiple-choice accuracy is the
opposite: it is exact, it cannot be gamed by paraphrase, and its ground truth is a
numeric label that survives translation untouched, which makes it the only metric
that is directly comparable across Spanish and Basque. It is defined only for
CasiMédicos-Exp records, so it is blank ("---") on the SNS-1064-only tables.

**The main text stages the comparison; the appendix shows everything.** Putting all
eleven configurations in one table asks the reader to hold eleven rows in mind and
find the contrasts themselves. Each main-text table instead answers one question,
carrying forward the winner of the previous stage as its reference row:

    Table 1  baseline        vs  dense retrieval (e5 top-1/3/5)   -> does retrieval help?
    Table 2  best dense      vs  reranking (top-1/3/5)            -> does the reranker help?
    Table 3  best reranked   vs  few-shot (alone, and combined)   -> format or knowledge?
    Table 4  best so far     vs  domain-restricted corpora        -> does the corpus matter?

The appendix retains all eleven rows for completeness, so nothing is hidden.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

METRICS_DIR = REPO / "reports" / "metrics"
OUT_DIR = REPO / "manuscript"
SEEDS = [42, 43, 44]

# (row label, config-id prefix, run-name base) per model, per language.
from write_mixed_es_seed_summary import EXPERIMENTS as ES_EXPERIMENTS  # noqa: E402
from write_mixed_eu_seed_summary import EXPERIMENTS as EU_EXPERIMENTS  # noqa: E402

# Quality columns. Cosim is deliberately absent (see module docstring).
QUALITY = [
    ("rouge_l_f1", r"ROUGE-L"),
    ("bertscore_f1", r"BERT-F1"),
    ("mc_accuracy", r"MC-acc"),
    ("meanq", r"MeanQ"),
]

# MeanQ is the mean of the three quality metrics (see scripts/meanq.py); it is the
# score the staged ablation uses to choose the best RAG configuration, so it is
# shown alongside its components. It is computed from the row's metrics, not read
# from a metric file. On SNS-1064 (open answer) MC-accuracy is undefined, so MeanQ
# there is the mean of the two overlap metrics.
MEANQ_COMPONENTS = ("rouge_l_f1", "bertscore_f1", "mc_accuracy")

# Raw metric fields collect() needs to pull from the metric JSONs (MeanQ is
# derived, computed separately below, not read from a file). The appendix used
# to additionally show cosine similarity, but it was dropped everywhere else
# (it saturates -- configurations separate by barely a point -- and scores a
# fluent, on-topic, factually wrong answer nearly as highly as a correct one),
# so the appendix now shows the same six metrics as the main text: no reason
# for the two to disagree on what "the" results are.
RAW_QUALITY_FIELDS = ("rouge_l_f1", "bertscore_f1", "mc_accuracy")

# Composite id: <experiment number><model letter>[']. The experiment number
# (0-10) identifies the row/condition -- baseline is 0, e5 top 1 is 1, ...
# domain-CasiMedicos is 10 -- assigned once from the canonical EXPERIMENTS list
# order (ES and EU share the same label text and order, so one map serves both).
# The model letter (a, b, c, ...) is assigned from that language's own model list
# order (ES_MODELS / EU_MODELS). A trailing ' marks the self-feedback row (e.g.
# "2a" noSF, "2a'" SF), so an id read in running prose is unambiguous on its own
# without also needing to say "SF"/"noSF" in words. Rows are additionally emitted
# model-major, SF-minor (2a, 2a', 2b, 2b', ...) so the two rows of one model sit
# adjacent -- see emit_table.
# Both numbers and letters are global across every table (staged + appendix): a
# row carried forward as a reference keeps the id it had when first introduced.
EXPERIMENT_NUMBERS: dict[str, int] = {}
MODEL_LETTERS: dict[str, str] = {}


def experiment_id(label: str, model: str, use_sf: bool) -> str:
    tick = "'" if use_sf else ""
    return f"{EXPERIMENT_NUMBERS[label]}{MODEL_LETTERS[model]}{tick}"


# The staged main-text tables. Each is (file slug, caption stem, question, row labels).
# The reference row of each stage is the winner of the previous one, resolved at
# build time rather than hard-coded, so the staging survives the numbers changing.
STAGES = [
    ("retrieval", "Effect of dense retrieval",
     "Does retrieval help at all, and how many passages?",
     ["Baseline LLM only", "e5 top 1", "e5 top 3", "e5 top 5"]),
    ("rerank", "Effect of cross-encoder reranking",
     "Does reranking a larger candidate pool beat dense retrieval alone?",
     ["rerank top 1", "rerank top 3", "rerank top 5"]),
    ("fewshot", "Effect of few-shot prompting",
     "Is the gain from demonstrating the output format, or from supplying knowledge?",
     ["3-shot, no RAG", "3-shot + rerank top 5"]),
    ("domain", "Effect of restricting the retrieval corpus",
     "Does retrieving from a single-domain corpus help or hurt?",
     ["cross-domain: SNS index", "cross-domain: CasiMedicos index"]),
]

# The corpus-restriction rows are not a different *method*, they are the best RAG
# configuration run against a corpus restricted to one dataset. Naming them
# "cross-domain: ..." obscured that; the label is rewritten at render time to say
# which corpus, and to carry the name of the configuration being held fixed.
DOMAIN_RENAME = {
    "cross-domain: SNS index": "SNS retrieval",
    "cross-domain: CasiMedicos index": "CasiMédicos retrieval",
}


def display_label(label: str, best_config: Optional[str]) -> str:
    if label in DOMAIN_RENAME:
        return f"{DOMAIN_RENAME[label]}, {best_config}" if best_config else DOMAIN_RENAME[label]
    return label


# Which rows each stage may pick its carried-forward reference from. Cumulative:
# each stage's pool is every row shown in this or an earlier stage, not just the
# stage immediately before it -- otherwise a genuinely best config (e.g. e5 top 5,
# which can beat every reranked row) could win one stage and then be structurally
# ineligible to be carried into the next, breaking the "reference = best system
# built so far" property the captions claim. "3-shot, no RAG" is deliberately
# excluded from the domain pool: it retrieves nothing, so it cannot be the RAG
# base a domain-\emph{restricted} row is built on (same reasoning as
# rewire_dependent_configs.py's staged-ablation pool for the same rows).
STAGE_POOL = {
    "rerank": ["e5 top 1", "e5 top 3", "e5 top 5"],
    "fewshot": ["e5 top 1", "e5 top 3", "e5 top 5",
                "rerank top 1", "rerank top 3", "rerank top 5"],
    "domain": ["e5 top 1", "e5 top 3", "e5 top 5",
               "rerank top 1", "rerank top 3", "rerank top 5",
               "3-shot + rerank top 5"],
}

# Manual tie-break: best_label() would pick "e5 top 3" for EU (MeanQ 47.64+/-3.59,
# the cross-model max -- Latxa's own best), but "e5 top 1" (47.17+/-2.87) is within
# noise of it, has a visibly tighter std, and costs roughly half the tokens/latency
# (see the EU retrieval-stage caption for the numbers). That is a judgment call
# ("mean+std+cost jointly favour the cheaper, more stable config"), not something
# a MeanQ-max rule should make automatically -- automating "prefer lower std when
# means are close" as a general rule risks silently overriding OTHER stages'
# references too. So this pins the EU reference from the rerank stage onward.
#
# ES's tie-break is a different shape: it is not between two labels but between
# two MODELS at the same label ("rerank top 5"). best_label()'s cross-model MeanQ
# max there is silently Qwen3.5-9B (think) (71.34+/-0.38), beating Qwen3.5-9B
# (no-think) (69.79+/-0.89) by 1.55 points -- a real gap, bigger than either std,
# so unlike EU this is NOT "within noise". The reason no-think is carried forward
# anyway is cost: think mode costs ~2.9x the wall-clock time and ~3.5x the tokens
# (2.03s/1854.59tok vs 5.96s/6541.49tok) for that 1.55-point gain, the same
# efficiency trade-off already made in the reasoning-pipeline section (see
# scripts/write_reasoning_latex_table.py's ES_ROWS comment). FORCED_MODEL pins
# which model's row counts as the winner at a forced label, so best_label() (and
# the reference-row lookup) resolve to no-think instead of the cross-model max.
FORCED_REFERENCE = {"EU": "e5 top 1", "ES": "rerank top 5"}
FORCED_MODEL = {"ES": "Qwen3.5-9B (no-think)"}

# EU's tie-break is between two LABELS but for a single model (Latxa): the
# highlighted MeanQ cells, and the auto-pick search used to build the tie-break
# note, must be restricted to Latxa alone -- otherwise Llama's own e5top1/e5top3
# rows (a different, unrelated comparison) get swept in too.
TIE_BREAK_MODEL = {"EU": "Latxa-8B"}


def esc(text: str) -> str:
    return (str(text).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def run_dir(prefix: str, base: str, seed: int) -> str:
    return f"{prefix}_{base}_seed{seed}"


def load_summary(run: str, suffix: str) -> Optional[dict[str, Any]]:
    path = METRICS_DIR / f"{run}{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("summary")


def values(summaries: list[dict], metric: str, *, use_sf: bool) -> list[float]:
    out = []
    for summary in summaries:
        block = "after_feedback" if use_sf else "before_feedback"
        value = (summary.get(block) or {}).get("overall", {}).get(metric)
        if value is None:
            value = (summary.get("overall") or {}).get(metric)
        if value is not None:
            out.append(float(value))
    return out


def value_or_none(summary: Optional[dict], metric: str, *, use_sf: bool) -> Optional[float]:
    """Like values(), but for one summary and keeping None instead of dropping it,
    so per-seed lists from different sources (e.g. mixed-file ROUGE-L/BERT-F1 vs.
    the _casimedicos-subset MC-acc) stay aligned by seed position when zipped."""
    if summary is None:
        return None
    block = "after_feedback" if use_sf else "before_feedback"
    value = (summary.get(block) or {}).get("overall", {}).get(metric)
    if value is None:
        value = (summary.get("overall") or {}).get(metric)
    return float(value) if value is not None else None


def mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def fmt(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "---"
    return f"{mean:.2f}{{\\tiny$\\pm${std:.2f}}}" if std else f"{mean:.2f}"


def _nested_mean(block: dict, name: str) -> float:
    value = (block.get(name) or {}).get("mean")
    return float(value) if value is not None else 0.0


def cost(summaries: list[dict], token: bool, *, use_sf: bool) -> Optional[float]:
    """noSF and SF cost are NOT the same number, even though the raw metric JSON
    only stores one pipeline-wide total (`example_seconds`, `total_tokens`) that
    already includes the self-feedback pass. That total IS the SF row's cost; the
    noSF row's cost has to be reconstructed by summing the pre-feedback components
    and excluding the feedback ones -- exactly the split
    scripts/summarize_metrics.py's cost_rows() already uses, so this mirrors it
    rather than inventing a second convention. Before this fix, cost() ignored
    use_sf entirely and both rows silently showed the SF (larger) total.
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


def collect(prefix: str, base: str, suffix: str, use_sf: bool) -> Optional[dict]:
    summaries = [s for s in (load_summary(run_dir(prefix, base, sd), suffix) for sd in SEEDS) if s]
    if not summaries:
        return None
    row: dict[str, Any] = {"n": len(summaries)}
    for metric in RAW_QUALITY_FIELDS:
        row[metric] = mean_std(values(summaries, metric, use_sf=use_sf))
    row["sec"] = cost(summaries, token=False, use_sf=use_sf)
    row["tok"] = cost(summaries, token=True, use_sf=use_sf)

    # MC-accuracy on the mixed table comes from the CasiMedicos subset (it is
    # undefined on the open-answer half, and the mixed-suffix metric files are
    # missing it outright for seeds other than 42 -- see the seed/prompt audit),
    # so it is read from the *_casimedicos metric files, matching scripts/meanq.py.
    # This OVERWRITES row["mc_accuracy"] (not just a local var used for MeanQ):
    # without that, the table's own MC-acc column silently fell back to the
    # partially-populated mixed-suffix value (often a single seed, so std
    # collapsed to 0.0 and fmt() dropped the +/- entirely). On tables that are
    # already CasiMedicos- or SNS-only, the row's own MC-accuracy value is used
    # (present or None respectively).
    mixed_summaries = [load_summary(run_dir(prefix, base, sd), suffix) for sd in SEEDS]
    if suffix == "":  # mixed table
        mc_summaries = [load_summary(run_dir(prefix, base, sd), "_casimedicos") for sd in SEEDS]
        row["mc_accuracy"] = mean_std(values([s for s in mc_summaries if s], "mc_accuracy", use_sf=use_sf))
    else:
        mc_summaries = mixed_summaries

    # MeanQ, per seed: pair that seed's ROUGE-L/BERT-F1 (mixed file) with that SAME
    # seed's MC-acc (casimedicos file for the mixed table, else the row's own file),
    # aligned by seed position (via value_or_none, which keeps a None placeholder
    # rather than silently dropping a missing seed and shifting later ones out of
    # alignment) -- then averaged over seeds -- exactly scripts/meanq.py's
    # meanq_per_seed method. Averaging three already-averaged component means (the
    # previous approach) has no per-seed variance to report, which is why MeanQ's
    # +/-std was always missing; this fixes that by construction.
    per_seed_meanq = []
    for i in range(len(SEEDS)):
        rouge = value_or_none(mixed_summaries[i], "rouge_l_f1", use_sf=use_sf)
        bert = value_or_none(mixed_summaries[i], "bertscore_f1", use_sf=use_sf)
        mc_seed = value_or_none(mc_summaries[i], "mc_accuracy", use_sf=use_sf)
        parts = [v for v in (rouge, bert, mc_seed) if v is not None]
        if parts:
            per_seed_meanq.append(sum(parts) / len(parts))
    row["meanq"] = mean_std(per_seed_meanq)
    return row


# ── models per language ───────────────────────────────────────────────────────
# Each EXPERIMENTS row is (label, id, base, id, base, ...) -- one (id, base) pair
# per model, in the order the decision tables report them.
ES_MODELS = ["Mistral-7B", "Qwen3.5-9B (no-think)", "Qwen3.5-9B (think)"]
EU_MODELS = ["Llama-3.1-8B", "Latxa-8B"]


def rows_for(experiments, models, label: str, suffix: str, use_sf: bool):
    """Return [(model_name, row_dict)] for one ablation condition."""
    entry = next((e for e in experiments if e[0] == label), None)
    if entry is None:
        return []
    rest = list(entry[1:])
    out = []
    for i, model in enumerate(models):
        if 2 * i + 1 >= len(rest):
            break
        prefix, base = rest[2 * i], rest[2 * i + 1]
        row = collect(prefix, base, suffix, use_sf)
        if row:
            out.append((model, row))
    return out


def best_label(experiments, models, pool, suffix, *, only_model: Optional[str] = None) -> Optional[tuple[str, str]]:
    """The (label, model) pair in `pool` with the highest MeanQ, on the
    no-self-feedback prediction -- the same criterion (metric and SF state) used
    everywhere else in the project to choose the best RAG base config (see
    scripts/meanq.py and reports/metrics/*_dev_ablation_results.md). Ranking by
    BERT-F1 alone, or searching over both SF states, could carry forward a
    different row than the one the rest of the project would call "best".

    Returns the winning MODEL alongside the label, not just the label: the
    reference carried into later stages is one specific config-model
    combination, not "this label, whichever model got there" -- showing every
    model's row under a label that only one of them actually won is misleading
    (see FORCED_MODEL's comment for the concrete case this caused).

    `only_model`, if given, restricts the search to that model's rows -- used to
    resolve FORCED_MODEL/TIE_BREAK_MODEL overrides, where the cross-model max at
    a label is not the model actually carried forward.

    Resolved from the data rather than hard-coded, so the staging still carries the
    right row forward when the numbers change under the seeded re-run.
    """
    best, best_score = None, float("-inf")
    for label in pool:
        for model, row in rows_for(experiments, models, label, suffix, use_sf=False):
            if only_model and model != only_model:
                continue
            mean = row["meanq"][0]
            if mean is not None and mean > best_score:
                best_score, best = mean, (label, model)
    return best


def emit_table(experiments, models, labels, *, caption, short, tag, suffix,
               best_config: Optional[str] = None, quality=None,
               highlight_rows: frozenset[tuple[str, str]] = frozenset(),
               restrict: Optional[dict[str, str]] = None,
               best_sf_only: frozenset[tuple[str, str]] = frozenset()) -> list[str]:
    """`labels` is the row labels to show, in order. `restrict`, if given, maps a
    label to the ONE model whose row should be shown for it -- used for a
    carried-forward reference row, which is one specific config-model
    combination, not every model's version of that label (a stage's own
    comparison labels are never in `restrict`, so they still show every model).

    `best_sf_only`, keyed the same way as `highlight_rows` ((label, model)
    pairs), collapses that row to whichever of noSF/SF has the higher MeanQ,
    instead of showing both -- used for the same carried-forward reference row
    and the tie-break comparison rows, where showing both SF states doubles the
    row count without adding to the point being made (which config/model wins,
    not whether self-feedback helps it)."""
    quality = quality or QUALITY
    restrict = restrict or {}
    # 4 label columns (#, Model, Experiment, SF) + quality + 2 cost columns.
    ncol = 4 + len(quality) + 2
    colspec = r"r l l c " + "c " * len(quality) + r"c c"
    header = (
        r"\# & Model & Experiment & SF & \multicolumn{%d}{c}{Quality $\uparrow$} & "
        r"\multicolumn{2}{c}{Cost $\downarrow$} \\" % len(quality)
        + "\n" + r"\cmidrule(lr){5-%d}\cmidrule(lr){%d-%d}" % (
            4 + len(quality), 5 + len(quality), ncol)
        + "\n" + r" &  &  &  & " + " & ".join(m for _, m in quality) + r" & sec & tok \\"
    )
    lines = [
        r"\begin{scriptsize}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{" + colspec + r"}",
        r"\caption[%s]{%s} \label{tab:%s} \\" % (short, caption, tag),
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
    def best_sf_state(nosf_row: Optional[dict], sf_row: Optional[dict]) -> bool:
        """Which of noSF/SF has the higher MeanQ, for best_sf_only rows -- True
        means SF wins. Falls back to whichever row exists if only one does."""
        if sf_row is None:
            return False
        if nosf_row is None:
            return True
        nosf_mean, sf_mean = nosf_row["meanq"][0], sf_row["meanq"][0]
        if nosf_mean is None:
            return True
        if sf_mean is None:
            return False
        return sf_mean > nosf_mean

    # Gather every row first, so the best value per metric can be found before
    # rendering. The best value is marked in BOLD, per metric column: the convention
    # in the literature, and unlike a shaded row it survives greyscale printing and
    # says which *metric* a configuration wins on, rather than implying it wins on all.
    gathered = []
    for label in labels:
        only_model = restrict.get(label)
        nosf_rows = dict(rows_for(experiments, models, label, suffix, False))
        sf_rows = dict(rows_for(experiments, models, label, suffix, True))
        for use_sf in (False, True):
            for model, row in rows_for(experiments, models, label, suffix, use_sf):
                if only_model and model != only_model:
                    continue
                if (label, model) in best_sf_only:
                    if use_sf != best_sf_state(nosf_rows.get(model), sf_rows.get(model)):
                        continue
                gathered.append((label, model, use_sf, row))

    best_of: dict[str, float] = {}
    for metric, _ in quality:
        vals = [r[metric][0] for _, _, _, r in gathered if r[metric][0] is not None]
        if vals:
            best_of[metric] = max(vals)

    for label in labels:
        # Model-major, SF-minor (2a, 2a', 2b, 2b', ...): the noSF and SF row of one
        # model sit adjacent, rather than every model's noSF row then every model's
        # SF row. rows_for() returns one use_sf slice at a time, in model order, so
        # both slices are fetched up front and then interleaved by model index.
        nosf_rows = dict(rows_for(experiments, models, label, suffix, False))
        sf_rows = dict(rows_for(experiments, models, label, suffix, True))
        only_model = restrict.get(label)
        for model in models:
            if only_model and model != only_model:
                continue
            keep_sf = best_sf_state(nosf_rows.get(model), sf_rows.get(model)) if (label, model) in best_sf_only else None
            for use_sf, rows_by_model in ((False, nosf_rows), (True, sf_rows)):
                if keep_sf is not None and use_sf != keep_sf:
                    continue
                row = rows_by_model.get(model)
                if row is None:
                    continue
                exp_cell = esc(display_label(label, best_config))
                cells = [
                    experiment_id(label, model, use_sf),
                    esc(model),
                    exp_cell,
                    r"\checkmark" if use_sf else "",
                ]
                for metric, _ in quality:
                    mean, std = row[metric]
                    cell = fmt(mean, std)
                    is_col_best = mean is not None and metric in best_of and mean == best_of[metric]
                    # highlight_rows marks a tie-break comparison's own MeanQ cell
                    # (not the whole row). The config being weighed is always the
                    # noSF one (best_label()'s search is noSF-only by convention),
                    # but best_sf_only may have collapsed this row down to its
                    # winning SF state -- so when that's the ONLY row shown for
                    # this (label, model), still highlight it rather than
                    # requiring noSF specifically. Guarded against double-wrapping
                    # when that cell is also the column's best value.
                    tie_break_sf_ok = not use_sf or (label, model) in best_sf_only
                    is_tie_break = metric == "meanq" and tie_break_sf_ok and (label, model) in highlight_rows
                    if is_col_best or is_tie_break:
                        cell = r"\textbf{%s}" % cell
                    cells.append(cell)
                cells += [
                    f"{row['sec']:.2f}" if row["sec"] is not None else "---",
                    f"{row['tok']:.0f}" if row["tok"] is not None else "---",
                ]
                lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [
        r"\end{longtable}",
        r"\end{scriptsize}",
    ]
    return lines


def assign_ids(experiments, models) -> None:
    """Assign the experiment number (0-10, canonical EXPERIMENTS order) and model
    letter (a, b, c, ..., canonical model-list order) once, before any table is
    rendered. If ids were instead allocated lazily as rows were drawn, the id a
    run received would depend on which table happened to be generated first, and
    the same experiment would carry different ids in the staged tables and the
    appendix.
    """
    for i, entry in enumerate(experiments):
        EXPERIMENT_NUMBERS[entry[0]] = i
    for i, model in enumerate(models):
        MODEL_LETTERS[model] = chr(ord("a") + i)


def build_language(experiments, models, lang: str, dev_slug: str, suffix: str) -> None:
    """Four staged main-text tables + one full appendix table, for one language/dataset."""
    # ── staged main-text tables ────────────────────────────────────────────────
    forced = FORCED_REFERENCE.get(lang)
    forced_model = FORCED_MODEL.get(lang)
    # If a manual override applies, resolve the tie-break note/highlight ONCE here,
    # against the pool where the forced and auto-picked configs first compete
    # (STAGE_POOL["rerank"], the set of rows the "retrieval" table displays, unless
    # the override is itself a rerank-stage label, in which case the comparison is
    # cross-model at that one label instead) -- rather than recomputing it per
    # stage, which would either miss the comparison (the retrieval table's own
    # `pool` is None, so best_label() is never called for it) or recompute it
    # redundantly for every later stage.
    #
    # Two override shapes are handled: EU picks between two LABELS for the same
    # model (e5 top 1 vs e5 top 3, both Latxa) -- the note stresses they are within
    # noise of each other. ES picks between two MODELS at the same label (Qwen
    # no-think vs think, both "rerank top 5") -- the gap there is real (bigger than
    # either std), so the note is cost-based instead: a genuine but small MeanQ
    # gain that isn't worth ~3x the compute.
    tie_break_note_slug = ""
    tie_break_note = ""
    highlight_rows: frozenset[tuple[str, str]] = frozenset()
    if forced_model:
        # Cross-model override at a single label: find which model would win it
        # absent the override, and compare against the forced model's own row.
        auto_pick_model = None
        best_score = float("-inf")
        for model, row in rows_for(experiments, models, forced, suffix, use_sf=False):
            mean = row["meanq"][0]
            if mean is not None and mean > best_score:
                best_score, auto_pick_model = mean, model
        if auto_pick_model and auto_pick_model != forced_model:
            forced_row = dict(rows_for(experiments, models, forced, suffix, use_sf=False))[forced_model]
            auto_row = dict(rows_for(experiments, models, forced, suffix, use_sf=False))[auto_pick_model]
            f_mean, f_std = forced_row["meanq"]
            a_mean, a_std = auto_row["meanq"]
            sec_ratio = auto_row["sec"] / forced_row["sec"] if forced_row["sec"] else None
            tok_ratio = auto_row["tok"] / forced_row["tok"] if forced_row["tok"] else None
            tie_break_note_slug = "rerank"
            highlight_rows = frozenset({(forced, forced_model), (forced, auto_pick_model)})
            tie_break_note = (
                f" MeanQ favours {auto_pick_model} over {forced_model} at {forced} "
                f"({a_mean:.2f}$\\pm${a_std:.2f} vs {f_mean:.2f}$\\pm${f_std:.2f}), a real "
                f"gain rather than noise, but it costs roughly {sec_ratio:.1f}$\\times$ the "
                f"latency and {tok_ratio:.1f}$\\times$ the tokens for it: {forced_model} is "
                f"carried forward into every later stage as the efficient choice."
            )
    elif forced:
        auto_pick_result = best_label(experiments, models, STAGE_POOL["rerank"], suffix, only_model=TIE_BREAK_MODEL.get(lang))
        auto_pick = auto_pick_result[0] if auto_pick_result else None
        if auto_pick and auto_pick != forced:
            tie_break_note_slug = "retrieval"
            pin_model = TIE_BREAK_MODEL.get(lang)
            highlight_rows = (
                frozenset({(forced, pin_model), (auto_pick, pin_model)}) if pin_model else
                frozenset((label, model) for label in (forced, auto_pick) for model in models)
            )
            tie_break_note = (
                f" MeanQ narrowly favours {auto_pick} over {forced}, but the two are "
                f"within a seed's worth of noise of each other, {forced} has the tighter "
                f"standard deviation, and it costs roughly half the tokens and latency: "
                f"{forced} is the config carried forward into every later stage."
            )

    # The forced reference's winning MODEL, resolved once: ES pins it directly
    # (FORCED_MODEL), EU's pin is Latxa specifically (TIE_BREAK_MODEL) -- both are
    # a single config-model combination, never "this label, any model".
    forced_pin_model = forced_model or TIE_BREAK_MODEL.get(lang)

    for slug, stem, question, labels in STAGES:
        pool = STAGE_POOL.get(slug)
        auto_reference = (
            best_label(experiments, models, pool, suffix, only_model=forced_model)
            if pool and forced_model else
            best_label(experiments, models, pool, suffix) if pool else None
        )
        # The override only applies once the forced config has actually appeared as
        # a candidate (from "rerank" onward -- "retrieval" has no reference row yet,
        # it's where e5 top 1/3 are first compared against each other). At the stage
        # where the tie-break comparison itself is being made (tie_break_note_slug),
        # the forced label is one of that stage's OWN rows, not a carried-forward
        # reference -- prepending it there would duplicate the row (ES: "rerank top
        # 5" is both the forced pick and a member of the "rerank" stage's own labels).
        is_tie_break_stage = slug == tie_break_note_slug and tie_break_note
        reference: Optional[tuple[str, str]] = None if is_tie_break_stage else (
            (forced, forced_pin_model) if (pool and forced) else auto_reference
        )
        ref_label = reference[0] if reference else None
        caption = (
            f"{stem} ({lang}, {dev_slug} dev). {question}"
            + (f" The reference row is the best configuration carried forward from the "
               f"previous stage ({ref_label}, {reference[1]}), so each comparison is "
               f"against the best system built so far. Best value per metric in bold."
               if reference else
               f" Best value per metric in bold.{tie_break_note}"
               if is_tie_break_stage else
               " Best value per metric in bold.")
        )
        shown = ([ref_label] if ref_label else []) + labels
        restrict = {ref_label: reference[1]} if reference else None
        # A carried-forward reference row shows only its better-MeanQ SF state --
        # both states would double the row without adding to what the table is
        # comparing (whether the OTHER conditions beat the reference, not whether
        # self-feedback helps the reference itself). The tie-break comparison rows
        # (highlight_rows, at their own stage) get the same treatment for the same
        # reason: EU's e5top1/top3 and ES's Qwen think/no-think are both already
        # pinned to noSF by convention (best_label()'s use_sf=False search), so
        # collapsing to the best SF state here reproduces that pin instead of
        # fighting it.
        stage_best_sf_only = (
            frozenset({(ref_label, reference[1])}) if reference else
            highlight_rows if slug == tie_break_note_slug else
            frozenset()
        )
        out = OUT_DIR / f"table_{lang.lower()}_{dev_slug}_{slug}.tex"
        out.write_text("\n".join(emit_table(
            experiments, models, shown,
            caption=caption, short=f"{stem} ({lang}, {dev_slug})",
            tag=f"{lang.lower()}-{dev_slug}-{slug}", suffix=suffix,
            best_config=ref_label,
            restrict=restrict,
            highlight_rows=highlight_rows if slug == tie_break_note_slug else frozenset(),
            best_sf_only=stage_best_sf_only,
        )) + "\n")

    # ── full appendix table: all eleven conditions ─────────────────────────────
    all_labels = [e[0] for e in experiments]
    out = OUT_DIR / f"appendix_table_{lang.lower()}_{dev_slug}.tex"
    out.write_text("\n".join(emit_table(
        experiments, models, all_labels,
        caption=(f"Full dev ablation ({lang}, {dev_slug}): all eleven configurations, "
                 f"each with and without self-feedback, mean$\\pm$std over seeds "
                 f"{', '.join(str(s) for s in SEEDS)}."),
        short=f"Full dev ablation ({lang}, {dev_slug})",
        tag=f"app-{lang.lower()}-{dev_slug}", suffix=suffix,
        quality=QUALITY,
    )) + "\n")


def main() -> None:
    for experiments, models, lang in (
        (ES_EXPERIMENTS, ES_MODELS, "ES"),
        (EU_EXPERIMENTS, EU_MODELS, "EU"),
    ):
        EXPERIMENT_NUMBERS.clear()   # ES and EU are independent grids, each numbered from 0
        MODEL_LETTERS.clear()
        assign_ids(experiments, models)
        for dev_slug, suffix in (("mixed", ""), ("sns1064", "_sns1064"),
                                 ("casimedicos", "_casimedicos")):
            build_language(experiments, models, lang, dev_slug, suffix)
    written = sorted(p.name for p in OUT_DIR.glob("table_*.tex")) + \
              sorted(p.name for p in OUT_DIR.glob("appendix_table_*.tex"))
    for name in written:
        print(f"  {name}")
    print(f"\n  {len(written)} table files written")


if __name__ == "__main__":
    main()
