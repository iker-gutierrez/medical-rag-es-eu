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


def cost(summaries: list[dict], field: str, token: bool) -> Optional[float]:
    vals = []
    for summary in summaries:
        block = (summary.get("cost") or {})
        block = (block.get("token_counts") if token else block.get("timing")) or {}
        value = (block.get(field) or {}).get("mean")
        if value is not None:
            vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def collect(prefix: str, base: str, suffix: str, use_sf: bool) -> Optional[dict]:
    summaries = [s for s in (load_summary(run_dir(prefix, base, sd), suffix) for sd in SEEDS) if s]
    if not summaries:
        return None
    row: dict[str, Any] = {"n": len(summaries)}
    for metric in RAW_QUALITY_FIELDS:
        row[metric] = mean_std(values(summaries, metric, use_sf=use_sf))
    row["sec"] = cost(summaries, "example_seconds", token=False)
    row["tok"] = cost(summaries, "total_tokens", token=True)

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


def best_label(experiments, models, pool, suffix) -> Optional[str]:
    """The label in `pool` with the highest MeanQ, on the no-self-feedback
    prediction, over any model -- the same criterion (metric and SF state) used
    everywhere else in the project to choose the best RAG base config (see
    scripts/meanq.py and reports/metrics/*_dev_ablation_results.md). Ranking by
    BERT-F1 alone, or searching over both SF states, could carry forward a
    different row than the one the rest of the project would call "best".

    Resolved from the data rather than hard-coded, so the staging still carries the
    right row forward when the numbers change under the seeded re-run.
    """
    best, best_score = None, float("-inf")
    for label in pool:
        for _, row in rows_for(experiments, models, label, suffix, use_sf=False):
            mean = row["meanq"][0]
            if mean is not None and mean > best_score:
                best_score, best = mean, label
    return best


def emit_table(experiments, models, labels, *, caption, short, tag, suffix,
               best_config: Optional[str] = None, quality=None) -> list[str]:
    quality = quality or QUALITY
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
    # Gather every row first, so the best value per metric can be found before
    # rendering. The best value is marked in BOLD, per metric column: the convention
    # in the literature, and unlike a shaded row it survives greyscale printing and
    # says which *metric* a configuration wins on, rather than implying it wins on all.
    gathered = []
    for label in labels:
        for use_sf in (False, True):
            for model, row in rows_for(experiments, models, label, suffix, use_sf):
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
        for model in models:
            for use_sf, rows_by_model in ((False, nosf_rows), (True, sf_rows)):
                row = rows_by_model.get(model)
                if row is None:
                    continue
                cells = [
                    experiment_id(label, model, use_sf),
                    esc(model),
                    esc(display_label(label, best_config)),
                    r"\checkmark" if use_sf else "",
                ]
                for metric, _ in quality:
                    mean, std = row[metric]
                    cell = fmt(mean, std)
                    if mean is not None and metric in best_of and mean == best_of[metric]:
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
    carried: Optional[str] = None
    for slug, stem, question, labels in STAGES:
        pool = STAGE_POOL.get(slug)
        reference = best_label(experiments, models, pool, suffix) if pool else None
        shown = ([reference] if reference else []) + labels
        caption = (
            f"{stem} ({lang}, {dev_slug} dev). {question}"
            + (f" The reference row is the best configuration carried forward from the "
               f"previous stage ({reference}), so each comparison is against the best "
               f"system built so far. Best value per metric in bold."
               if reference else " Best value per metric in bold.")
        )
        out = OUT_DIR / f"table_{lang.lower()}_{dev_slug}_{slug}.tex"
        out.write_text("\n".join(emit_table(
            experiments, models, shown,
            caption=caption, short=f"{stem} ({lang}, {dev_slug})",
            tag=f"{lang.lower()}-{dev_slug}-{slug}", suffix=suffix,
            best_config=reference,
        )) + "\n")
        carried = reference

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
