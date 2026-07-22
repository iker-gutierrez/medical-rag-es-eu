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

    Table 1  baseline        vs  dense retrieval (retrieve top-1/3/5)   -> does retrieval help?
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
# (0-10) identifies the row/condition -- baseline is 0, retrieve top 1 is 1, ...
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
     ["Baseline LLM only", "retrieve top 1", "retrieve top 3", "retrieve top 5"]),
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

# The domain-restriction rows' base config PER MODEL, as an independent fact --
# NOT derived from FORCED_REFERENCES, because the domain runs are fixed
# experiments (already run against one specific base per model) and don't move
# when the pinned reference list changes. Checked directly against the actual
# config JSONs rather than assumed from filenames (which can be stale):
#
# ES (configs/experiments/1136_qwen35_9b_rag_sns1064_...,
# 1268_mistral7b_rag_sns1064_..., 1147/1278_..._think_...) all have
# reranker_top_k=5 and rag_base_source=None, confirming "rerank top 5" as
# filenamed, for all three models -- matches each model's own pin, no rerun
# needed.
#
# EU (full staged rerun against the rebuilt CasiMedicos-Exp/SNS-1064 splits
# and rebuilt retrieval indices, 2026-07-21): both Llama's and Latxa's domain
# rows were run against "retrieve top 3" as their base (dense-only, no
# reranking) -- see FORCED_REFERENCES below for the per-model rationale.
DOMAIN_BASE_LABEL: dict[str, dict[str, str]] = {
    # Qwen think's domain rows (1278/1279) were rebuilt with few_shot_k=3 added
    # on top of rerank top 5, i.e. row 8's "3-shot + rerank top 5" base, per
    # explicit request; Qwen no-think's domain rows (1136/1137) were not
    # touched and remain on their original rerank-top-5 base.
    "ES": {"Qwen no-think": "rerank top 5", "Qwen think": "3-shot + rerank top 5"},
    "EU": {"Llama": "retrieve top 3", "Latxa": "retrieve top 3"},
}


def display_label(label: str, best_config: Optional[str], model: str) -> str:
    if label in DOMAIN_RENAME:
        config_text = best_config.get(model) if isinstance(best_config, dict) else best_config
        return f"{DOMAIN_RENAME[label]}, {config_text}" if config_text else DOMAIN_RENAME[label]
    if label == "3-shot + rerank top 5":
        # This row's RAG base isn't fixed at "rerank top 5" -- it's rewired to
        # whichever config wins stage A's MeanQ (see run_model()'s apply_base
        # call), so the static label goes stale whenever that winner changes
        # (as it did: rerank5 -> retrieve top3 once MC-acc was correctly
        # included). Substitute the actual pinned config so the table never
        # names a RAG setting the row wasn't actually run with.
        config_text = best_config.get(model) if isinstance(best_config, dict) else best_config
        # Guard against self-substitution: when THIS row's own literal label
        # (e.g. row 8, carried forward as the domain stage's reference) is
        # itself "3-shot + rerank top 5" and that also happens to be the
        # pinned config_text (ES_STAGE_REFERENCE_OVERRIDE's domain entry),
        # naively prepending would double the prefix ("3-shot + 3-shot +
        # rerank top 5"). Row 8 already names its own base; leave it as-is.
        if config_text == label:
            return label
        return f"3-shot + {config_text}" if config_text else label
    return label


# Which rows each stage may pick its carried-forward reference from. Cumulative:
# each stage's pool is every row shown in this or an earlier stage, not just the
# stage immediately before it -- otherwise a genuinely best config (e.g. retrieve top 5,
# which can beat every reranked row) could win one stage and then be structurally
# ineligible to be carried into the next, breaking the "reference = best system
# built so far" property the captions claim. "3-shot, no RAG" is deliberately
# excluded from the domain pool: it retrieves nothing, so it cannot be the RAG
# base a domain-\emph{restricted} row is built on (same reasoning as
# rewire_dependent_configs.py's staged-ablation pool for the same rows).
STAGE_POOL = {
    "rerank": ["retrieve top 1", "retrieve top 3", "retrieve top 5"],
    "fewshot": ["retrieve top 1", "retrieve top 3", "retrieve top 5",
                "rerank top 1", "rerank top 3", "rerank top 5"],
    "domain": ["retrieve top 1", "retrieve top 3", "retrieve top 5",
               "rerank top 1", "rerank top 3", "rerank top 5",
               "3-shot + rerank top 5"],
}

# Manual pins, chosen PER MODEL (each model carries forward its OWN best config,
# not a single config shared across every model in the language) -- one pin per
# model, except ES's Qwen no-think/think tie (see below, still two pins because
# they're a genuine unresolved trade-off, not a per-model split).
#
# EU (full staged rerun against the rebuilt CasiMedicos-Exp/SNS-1064 splits and
# rebuilt retrieval indices, 2026-07-21). Chosen by scripts/meanq.py's
# best_by_meanq_robust(): highest mean MeanQ, UNLESS a candidate within 0.5
# MeanQ of the leader has meaningfully lower std (<=0.6x the leader's) or
# meaningfully lower cost (<=0.6x, cost = retrieval_top_k, +5 if reranked) --
# the reader's explicit variance/cost-aware selection rule, not a plain MeanQ
# argmax. This is the reference CARRIED FORWARD into stage 3 (few-shot) and
# stage 4 (domain) -- those rows were actually generated against this base:
#   Llama-3.1-8B -> retrieve top 3, no self-feedback (49.01+/-1.71) -- the
#   outright highest MeanQ Llama reaches across every retrieval AND rerank
#   candidate (next: retrieve top1 46.81+/-0.84, retrieve top5 46.61+/-1.62,
#   rerank5 46.24+/-1.01, rerank1 43.88+/-0.85, rerank3 43.12+/-1.00). A clear
#   leader by margin alone; the std/cost tiebreak never needs to engage.
#   Latxa-8B -> retrieve top 3, no self-feedback (47.28+/-0.73) -- narrowly
#   beats the next candidate, rerank top 3 (46.81+/-1.03), by only 0.47 MeanQ,
#   inside the 0.5-point margin. Std alone would NOT have triggered a
#   stability override (0.73 vs 0.6x1.03=0.62), but retrieve top 3's cost
#   (3, dense-only) is well under 0.6x rerank top 3's cost (20, reranked),
#   so the cost tiebreak decides it: cheap dense retrieval over an expensive
#   reranked config that isn't reliably better. (Other candidates, for
#   completeness: retrieve top5 45.63+/-0.51, rerank5 45.26+/-2.06, retrieve
#   top1 41.72+/-0.59, rerank1 40.22+/-0.61.)
#   Both models land on the SAME label this rerun (retrieve top 3) -- unlike
#   the previous rerun, where they pinned at different labels (rerank top 5 /
#   retrieve top 5) and each pin's own stage had to be derived independently
#   (see pin_own_index() in build_language(), still needed in general even
#   though it resolves to the same stage for both here).
#
# ES (updated 2026-07-21 for the rebuilt data/indices; Mistral-7B-Instruct-v0.3
# dropped from the roster, superseded by the Ministral-3-8B models, whose own
# staged rerun was still in progress as of this update and is not yet reflected
# here): Qwen no-think's own best config is confirmed "rerank top 5" against
# fresh predictions (best_by_meanq_robust: 69.15+/-0.72 vs 2nd-best rerank3
# 67.12+/-1.55, well outside the variance/cost margin -- a clean win, no
# tiebreak needed). Qwen think's pin below (rerank top 5, 71.34+/-0.38 vs
# rerank3 70.41+/-0.61) is CARRIED FORWARD from its pre-rebuild run -- its own
# staged rerun against the rebuilt data was still running as of this update and
# has not yet been re-verified; re-check with best_by_meanq_robust once it
# completes and update this pin (and the MeanQ figures in the tie-break note
# below) if the winning label or numbers change.
FORCED_REFERENCES: dict[str, list[tuple[str, str]]] = {
    "EU": [("retrieve top 3", "Llama"), ("retrieve top 3", "Latxa")],
    "ES": [("retrieve top 1", "Qwen no-think"), ("retrieve top 5", "Qwen think")],
}

# ES manual override, by explicit user request: rather than one reference per
# model carried unchanged through every later stage (FORCED_REFERENCES's
# default semantics), BOTH models' OWN reference is updated stage by stage --
# each later table's "best system built so far" is a different, more recent
# row per model. This does not reflect an automatic MeanQ selection at every
# entry (see best_by_meanq_robust in meanq.py for the actual automatic pick
# at each stage); several entries are deliberate manual choices, overriding
# FORCED_REFERENCES's stage-1 entries for every stage from "rerank" onward.
#
# Qwen no-think: stage-1's own pin was retrieve top 1 (65.76 MeanQ, that
# stage's own winner), but from stage 2 onward the reference is rerank top 5
# (6a, 69.15 MeanQ) -- Qwen no-think's actual best-so-far row once reranking
# is introduced, clearly ahead of retrieve top 1, per explicit user request
# (previously left at retrieve top 1 through an oversight; corrected here).
#
# Keyed (stage_slug) -> the (label, model) pairs that REPLACE, not add to,
# the reference list computed from FORCED_REFERENCES for that stage. Stage
# "retrieval" itself is absent (no override there: FORCED_REFERENCES's own
# pin-not-yet-reached logic already shows no carried-forward reference in a
# pin's own stage).
ES_STAGE_REFERENCE_OVERRIDE: dict[str, list[tuple[str, str]]] = {
    "rerank": [("rerank top 5", "Qwen no-think"), ("rerank top 5", "Qwen think")],
    "fewshot": [("rerank top 5", "Qwen no-think"), ("rerank top 5", "Qwen think")],
    "domain": [("rerank top 5", "Qwen no-think"), ("3-shot + rerank top 5", "Qwen think")],
}

# Which SF state to highlight for a manually-forced pin, when it differs from
# the automatic higher-MeanQ choice best_sf_state() would make. Qwen think's
# rerank-top-5 row (6b/6b') is one case: SF's MeanQ (72.24) is actually
# marginally higher than noSF's (72.13), so the automatic rule would highlight
# 6b', but the request was to highlight 6b (noSF) specifically and remove
# 6b''s highlight -- both rows remain visible in the "rerank" stage's own
# table (this is not best_sf_only, which would hide one state entirely), only
# which one gets \rowcolor{pinnedrow} and the bold MeanQ changes. Same
# reasoning applies to Qwen think's stage-1 pin (3b/3b'): SF's raw MeanQ
# (69.93) edges out noSF's (69.90), but the request explicitly said
# "bluehighlight ... 3b", so noSF is forced here too, in the "retrieval"
# stage itself (this stage's own pin, not a later carried-forward reference).
# Keyed (stage_slug, label, model) -> False means noSF, True means SF. Note
# the SAME (label, model) pin needs an entry PER STAGE it's rendered in: its
# own stage (e.g. "rerank" for 6b/6b') AND every later stage where it appears
# as the carried-forward reference row (e.g. "fewshot", where stage 3's table
# shows Qwen think's rerank-top-5 reference row and must also force noSF
# there, or best_sf_state()'s automatic pick would silently reassert 6b').
ES_PIN_SF_OVERRIDE: dict[tuple[str, str, str], bool] = {
    ("retrieval", "retrieve top 5", "Qwen think"): False,
    ("rerank", "rerank top 5", "Qwen think"): False,
    ("fewshot", "rerank top 5", "Qwen think"): False,
}
# Which (label, model) pairs, if any, get the tie-break caption note and MeanQ
# highlight at their pin's own stage -- for a pair sharing ONE label between
# two models, worth explaining as a real trade-off. Empty now: ES's stage-1
# pins are two DIFFERENT labels (retrieve top 1 for Qwen no-think, retrieve
# top 5 for Qwen think, a manual per-model choice -- see FORCED_REFERENCES),
# not a same-label split, so the tie-break framing this mechanism renders no
# longer applies; EU's two pins were already a clean single winner each with
# nothing to explain. Kept as a mechanism (not deleted) in case a future
# rerun reintroduces a genuine same-label two-model split.
TIE_BREAK_PAIRS: dict[str, list[tuple[str, str]]] = {}
TIE_BREAK_STAGE: dict[str, str] = {}

# Extra manual blue-highlights, scoped to ONE stage rather than carried
# forward as a reference row everywhere (unlike FORCED_REFERENCES/pins).
# Keyed (lang, stage_slug) -> set of (label, model) pairs. Used when a model's
# own best-in-stage-1 row differs from the label carried forward into later
# stages (e.g. previously, EU's Llama: stage 1's own winner was retrieve top 5,
# but the carried-forward pin was rerank top 5 -- both needed separate
# highlighting).
#
# ES's "rerank" entry: both Qwen think's rerank-top-5 row (6b) and Qwen
# no-think's rerank-top-5 row (6a) are the winners carried forward into stage
# 3 (fewshot) via ES_STAGE_REFERENCE_OVERRIDE, but that carry-forward only
# highlights the row in LATER stages' tables -- within stage 2's OWN table,
# both 6a and 6b need their own explicit highlight too, per request.
EXTRA_PIN_ROWS: dict[tuple[str, str], frozenset[tuple[str, str]]] = {
    ("ES", "rerank"): frozenset({("rerank top 5", "Qwen think"), ("rerank top 5", "Qwen no-think")}),
}


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
ES_MODELS = ["Qwen no-think", "Qwen think"]
EU_MODELS = ["Llama", "Latxa"]


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
               best_config: Optional[str] | Optional[dict[str, str]] = None, quality=None,
               highlight_rows: frozenset[tuple[str, str]] = frozenset(),
               restrict: Optional[dict[str, frozenset[str]]] = None,
               best_sf_only: frozenset[tuple[str, str]] = frozenset(),
               pin_rows: frozenset[tuple[str, str]] = frozenset(),
               pin_sf_override: Optional[dict[tuple[str, str], bool]] = None,
               color_pins: bool = True,
               separator_after: int = 0) -> list[str]:
    """`labels` is the row labels to show, in order. `restrict`, if given, maps a
    label to the SET of models whose rows should be shown for it -- used for
    carried-forward reference row(s), which are specific config-model
    combinations, not every model's version of that label (a stage's own
    comparison labels are never in `restrict`, so they still show every model).
    Usually one model, but a tie-break carrying both winners forward (e.g. ES's
    Qwen no-think AND think at "rerank top 5") puts two in the same label's set.

    `best_sf_only`, keyed the same way as `highlight_rows` ((label, model)
    pairs), collapses that row to whichever of noSF/SF has the higher MeanQ,
    instead of showing both -- used for the same carried-forward reference row(s)
    and the tie-break comparison rows, where showing both SF states doubles the
    row count without adding to the point being made (which config/model wins,
    not whether self-feedback helps it).

    `pin_rows`, keyed like `highlight_rows`, marks a row with a translucent blue
    background (\\rowcolor{pinnedrow}): that model's OWN best config, wherever it
    appears -- as one of a stage's own comparison rows (marking which one wins
    and will be carried forward) or as a later stage's reference row (marking
    that it still is that model's best). `separator_after`, if nonzero, draws a
    dashed rule after that many leading reference-row labels, visually splitting
    "carried forward from earlier stages" from "new this stage" without the
    reader needing to parse the caption."""
    quality = quality or QUALITY
    restrict = restrict or {}
    # 4 label columns (#, Model, Config, SF) + quality + 2 cost columns.
    ncol = 4 + len(quality) + 2
    colspec = r"r l l c " + "c " * len(quality) + r"c c"
    header = (
        r"\# & Model & Config & SF & \multicolumn{%d}{c}{Quality $\uparrow$} & "
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
        allowed_models = restrict.get(label)
        nosf_rows = dict(rows_for(experiments, models, label, suffix, False))
        sf_rows = dict(rows_for(experiments, models, label, suffix, True))
        for use_sf in (False, True):
            for model, row in rows_for(experiments, models, label, suffix, use_sf):
                if allowed_models and model not in allowed_models:
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
    # MeanQ is bolded PER MODEL, not once globally: each model carries forward
    # its OWN best config into the next stage (see FORCED_REFERENCES's
    # per-model pins), so the reader needs to see which row is best for Llama
    # and, separately, which is best for Latxa -- a single global-max bold
    # would only ever mark one model's row and leave the other model's own
    # winner unmarked.
    best_meanq_per_model: dict[str, float] = {}
    for _, model, _, r in gathered:
        mean = r["meanq"][0]
        if mean is not None and mean > best_meanq_per_model.get(model, float("-inf")):
            best_meanq_per_model[model] = mean

    for label_index, label in enumerate(labels):
        # Model-major, SF-minor (2a, 2a', 2b, 2b', ...): the noSF and SF row of one
        # model sit adjacent, rather than every model's noSF row then every model's
        # SF row. rows_for() returns one use_sf slice at a time, in model order, so
        # both slices are fetched up front and then interleaved by model index.
        nosf_rows = dict(rows_for(experiments, models, label, suffix, False))
        sf_rows = dict(rows_for(experiments, models, label, suffix, True))
        allowed_models = restrict.get(label)
        for model in models:
            if allowed_models and model not in allowed_models:
                continue
            if (label, model) in best_sf_only:
                if (label, model) in (pin_sf_override or {}):
                    keep_sf = pin_sf_override[(label, model)]
                else:
                    keep_sf = best_sf_state(nosf_rows.get(model), sf_rows.get(model))
            else:
                keep_sf = None
            # Which SF state to paint blue for a pinned (label, model): whichever
            # has the higher MeanQ, same rule as best_sf_only -- but computed
            # independently of it, since a pin's OWN stage still shows BOTH SF
            # states (best_sf_only is empty there) while only one of them should
            # get the row highlight.
            if (label, model) in (pin_sf_override or {}):
                pin_sf = pin_sf_override[(label, model)]
            elif (label, model) in pin_rows:
                pin_sf = best_sf_state(nosf_rows.get(model), sf_rows.get(model))
            else:
                pin_sf = None
            for use_sf, rows_by_model in ((False, nosf_rows), (True, sf_rows)):
                if keep_sf is not None and use_sf != keep_sf:
                    continue
                row = rows_by_model.get(model)
                if row is None:
                    continue
                exp_cell = esc(display_label(label, best_config, model))
                is_pinned_row = (label, model) in pin_rows and (pin_sf is None or use_sf == pin_sf)
                row_prefix = r"\rowcolor{pinnedrow}" if (is_pinned_row and color_pins) else ""
                cells = [
                    experiment_id(label, model, use_sf),
                    esc(model),
                    exp_cell,
                    r"\checkmark" if use_sf else "",
                ]
                for metric, _ in quality:
                    mean, std = row[metric]
                    cell = fmt(mean, std)
                    if metric == "meanq":
                        is_col_best = mean is not None and mean == best_meanq_per_model.get(model)
                    else:
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
                    # Every blue-bar (pinned) row also gets its MeanQ bolded --
                    # the row color already marks "this model's chosen config",
                    # bolding MeanQ ties that back to the number that made it
                    # the choice, not just a whole-row splash of color.
                    is_pin_meanq = metric == "meanq" and is_pinned_row
                    if is_col_best or is_tie_break or is_pin_meanq:
                        cell = r"\textbf{%s}" % cell
                    cells.append(cell)
                cells += [
                    f"{row['sec']:.2f}" if row["sec"] is not None else "---",
                    f"{row['tok']:.0f}" if row["tok"] is not None else "---",
                ]
                lines.append(row_prefix + " & ".join(cells) + r" \\")
        # Dashed rule after the last reference-row label, splitting "carried
        # forward from earlier stages" from "new this stage" -- separator_after
        # counts LABELS, not rendered rows, since a reference label can itself
        # expand to several rows (multiple models, or noSF+SF). Extra space on
        # BOTH sides (4pt, vs the normal 2pt between row groups) so the rule
        # reads as a deliberate section break rather than just another gap --
        # \cdashline leaves no vertical gap of its own on either side.
        if separator_after and label_index == separator_after - 1:
            lines.append(r"\addlinespace[4pt]")
            lines.append(r"\cdashline{1-%d}" % ncol)
            lines.append(r"\addlinespace[4pt]")
        else:
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


def build_language(experiments, models, lang: str, dev_slug: str, suffix: str, color_pins: bool = True) -> None:
    """Four staged main-text tables + one full appendix table, for one language/dataset."""
    # ── staged main-text tables ────────────────────────────────────────────────
    pins = FORCED_REFERENCES.get(lang, [])
    tie_break_stage = TIE_BREAK_STAGE.get(lang)
    tie_break_pairs = TIE_BREAK_PAIRS.get(lang, [])
    tie_break_note = ""
    highlight_rows: frozenset[tuple[str, str]] = frozenset()
    if tie_break_stage and len(tie_break_pairs) == 2:
        (label1, model1), (label2, model2) = tie_break_pairs
        row1 = dict(rows_for(experiments, models, label1, suffix, use_sf=False)).get(model1)
        row2 = dict(rows_for(experiments, models, label2, suffix, use_sf=False)).get(model2)
        highlight_rows = frozenset({(label1, model1), (label2, model2)})
        if row1 and row2:
            m1, s1 = row1["meanq"]
            m2, s2 = row2["meanq"]
            if label1 == label2:
                # Same label, two models: the gap is real cost-vs-quality, not
                # noise -- report it as a trade-off, not a coin flip.
                sec_ratio = row2["sec"] / row1["sec"] if row1["sec"] else None
                tok_ratio = row2["tok"] / row1["tok"] if row1["tok"] else None
                tie_break_note = (
                    f" MeanQ favours {model2} over {model1} at {label1} "
                    f"({m2:.2f}$\\pm${s2:.2f} vs {m1:.2f}$\\pm${s1:.2f}), a real gain "
                    f"rather than noise, but it costs roughly {sec_ratio:.1f}$\\times$ the "
                    f"latency and {tok_ratio:.1f}$\\times$ the tokens for it: both are "
                    f"carried forward into every later stage rather than picking one."
                )
            else:
                # Two labels, same model: within noise of each other.
                tie_break_note = (
                    f" MeanQ narrowly favours {label2} over {label1} for {model1}, but the "
                    f"two are within a seed's worth of noise of each other: both are "
                    f"carried forward into every later stage rather than picking one."
                )

    stage_slugs = [s[0] for s in STAGES]
    stage_labels = {s[0]: s[3] for s in STAGES}
    # Each pin's own stage is derived, not hand-specified: whichever stage's
    # OWN `labels` contains that pin's label. Pins are no longer required to
    # share one "own stage" per language -- e.g. EU can pin Llama at a rerank
    # label and Latxa at a retrieval label simultaneously; each is only
    # prepended as a carried-forward reference from ITS OWN stage onward.
    def pin_own_index(pin_label: str) -> int:
        for idx, slug in enumerate(stage_slugs):
            if pin_label in stage_labels[slug]:
                return idx
        return -1
    for stage_index, (slug, stem, question, labels) in enumerate(STAGES):
        is_tie_break_stage = slug == tie_break_stage
        # A pin only applies (is prepended as a carried-forward reference) from
        # ITS OWN stage onward -- at its own stage, the pinned label is already
        # one of that stage's own rows, so prepending it would duplicate it;
        # every LATER stage gets it prepended, since the pipeline has by then
        # moved past the point where that config was chosen. Each pin is
        # checked independently, since different pins can now have different
        # own stages.
        reference = [(lbl, mdl) for lbl, mdl in pins if stage_index > pin_own_index(lbl)]
        if lang == "ES" and slug in ES_STAGE_REFERENCE_OVERRIDE:
            # Same "own stage" guard as the static `pins` above: an override
            # entry whose label belongs to THIS stage (e.g. Qwen no-think's
            # rerank-stage override is itself "rerank top 5", a rerank-stage
            # row) is already one of this stage's own comparison rows, so
            # showing it again as a carried-forward reference would duplicate
            # it. Only override entries whose own stage is earlier are kept.
            reference = [
                (lbl, mdl) for lbl, mdl in ES_STAGE_REFERENCE_OVERRIDE[slug]
                if stage_index > pin_own_index(lbl)
            ]
        ref_labels = sorted({label for label, _ in reference})
        ref_items = [f"{lbl}, {mdl}" for lbl, mdl in reference]
        if len(ref_items) <= 2:
            ref_desc = " and ".join(ref_items)
        else:
            ref_desc = ", ".join(ref_items[:-1]) + f", and {ref_items[-1]}"
        row_word = "row is the best configuration" if len(reference) == 1 else "rows are the best configurations"
        system_word = "the best system built" if len(reference) == 1 else "the best systems built"
        caption = (
            f"{stem} ({lang}, {dev_slug} dev). {question}"
            + (f" The reference {row_word} carried forward from the previous "
               f"stage ({ref_desc}), so each comparison is against {system_word} "
               f"so far. Best value per metric in bold."
               if reference else
               f" Best value per metric in bold.{tie_break_note}"
               if is_tie_break_stage and tie_break_note else
               " Best value per metric in bold.")
        )
        shown = ref_labels + labels
        # restrict maps a label to the SET of models allowed for it -- both pins
        # may share a label (ES) or not (EU), so this groups by label rather than
        # assuming one model per label.
        restrict: dict[str, frozenset[str]] = {}
        for lbl, mdl in reference:
            restrict[lbl] = restrict.get(lbl, frozenset()) | {mdl}
        # A carried-forward reference row shows only its better-MeanQ SF state --
        # both states would double the row without adding to what a LATER stage's
        # table is comparing (whether the OTHER conditions beat the references,
        # not whether self-feedback helps a reference itself).
        stage_best_sf_only = frozenset(reference)
        # The domain stage's row-suffix text ("SNS retrieval, X") names the fixed
        # base the domain runs were actually built on (DOMAIN_BASE_LABEL), not
        # whichever reference labels happen to be pinned -- the domain
        # experiments don't move when the pinned list changes (see its comment).
        domain_base = DOMAIN_BASE_LABEL.get(lang) if slug == "domain" else None
        # Per-model pinned-label dict for display_label()'s "3-shot + <base>"
        # substitution: NOT ref_labels[0] (which would collapse to None
        # whenever the pins span more than one distinct label -- e.g. a
        # previous EU rerun had Llama pinned at rerank top 5 while Latxa was
        # pinned at retrieve top 5; both EU pins currently share one label,
        # retrieve top 3, but the per-model dict is kept general rather than
        # assuming that stays true) -- built the same way DOMAIN_BASE_LABEL
        # already is, one label per model, so each model's row 8 names the
        # RAG base IT was actually built on.
        fewshot_base = {mdl: lbl for lbl, mdl in reference} if reference else None
        extra_pins = EXTRA_PIN_ROWS.get((lang, slug), frozenset())
        # pin_rows drives the blue \rowcolor{pinnedrow} highlight. It normally
        # tracks the same static `pins` used to compute `reference` above, but
        # when ES_STAGE_REFERENCE_OVERRIDE has replaced `reference` for this
        # stage, the highlight must track the OVERRIDE too, or the blue-marked
        # row and the row actually shown/carried-forward would disagree.
        this_stage_pins = (
            ES_STAGE_REFERENCE_OVERRIDE[slug] if (lang == "ES" and slug in ES_STAGE_REFERENCE_OVERRIDE)
            else pins
        )
        pin_sf_override = {
            (lbl, mdl): sf for (stg, lbl, mdl), sf in ES_PIN_SF_OVERRIDE.items()
            if lang == "ES" and stg == slug
        }
        out = OUT_DIR / f"table_{lang.lower()}_{dev_slug}_{slug}.tex"
        out.write_text("\n".join(emit_table(
            experiments, models, shown,
            caption=caption, short=f"{stem} ({lang}, {dev_slug})",
            tag=f"{lang.lower()}-{dev_slug}-{slug}", suffix=suffix,
            best_config=domain_base if slug == "domain" else fewshot_base,
            restrict=restrict or None,
            highlight_rows=highlight_rows if is_tie_break_stage else frozenset(),
            best_sf_only=stage_best_sf_only,
            pin_rows=frozenset(this_stage_pins) | extra_pins,
            pin_sf_override=pin_sf_override or None,
            color_pins=color_pins,
            separator_after=len(ref_labels) if reference else 0,
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
