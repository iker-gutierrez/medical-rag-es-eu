"""Generate LaTeX longtable appendix files for ES and EU dev ablation results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from write_supervisor_summary import (
    QUALITY_METRICS, QUALITY_METRIC_LABELS, SECTIONS, SECTION_LABELS,
    fmt, no_sf_metrics, delta_metrics, no_sf_cost, sf_cost, delta_cost,
)
from update_es_ablation_with_qwen3 import (
    MISTRAL_RUNS, QWEN_EXP_LABELS, QWEN_SLUGS, QWEN_DEV_IDS,
    QWEN_CONFIG_IDX, THINK_OFFSET,
    load_summary, after_feedback_metrics, after_feedback_cost,
    mistral_rows, qwen_rows, _best_key_for,
)
from write_eu_ablation_summary import (
    MODEL_ORDER, DEV_SLUGS, DEV_LABELS, DEV_SIZES,
    EXPERIMENT_ORDER,
    build_all_rows as eu_build_all_rows,
)

METRICS_DIR = REPO / "reports/metrics"
OUT_DIR = REPO / "manuscript"
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    """Escape LaTeX special characters in a string."""
    return (str(s)
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
            .replace("☞", r"$\Rightarrow$")
            .replace("Δ", r"$\Delta$")
            .replace("~", r"\textasciitilde{}"))

def fmtv(v) -> str:
    if v is None:
        return "---"
    try:
        return f"{float(v):.2f}"
    except (ValueError, TypeError):
        return str(v)

# ── ES tables ─────────────────────────────────────────────────────────────────

# Columns: #, model, experiment, think, SF, then 8 metrics × 3 sections, then 4 cost cols
# That's 5 + 24 + 4 = 33 columns — too wide. We'll use overall section only for quality
# (8 metrics × 1 section = 8) plus cost = 5 + 8 + 4 = 17 columns. Still wide.
# Use landscape longtable with footnotesize.

ES_QUALITY_COLS = list(QUALITY_METRICS)   # 8 metrics, overall section only
ES_COL_LABELS   = [QUALITY_METRIC_LABELS[m] for m in ES_QUALITY_COLS]

def es_col_spec() -> str:
    # #, model, experiment, think, SF = 5 text cols; 8 metric + 4 cost = 12 numeric
    return r"l l l l l " + " ".join(["r"] * (len(ES_QUALITY_COLS) + 4))

def es_header(caption: str, label: str) -> list[str]:
    metric_heads = " & ".join(esc(h) for h in ES_COL_LABELS)
    return [
        r"\begin{landscape}",
        r"\begin{footnotesize}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{es_col_spec()}}}",
        rf"\caption{{{esc(caption)}}} \label{{{label}}} \\",
        r"\toprule",
        r"\# & Model & Experiment & Think & SF & "
        + metric_heads
        + r" & sec & inp.tok & out.tok & tot.tok \\",
        r"\midrule",
        r"\endfirsthead",
        "\\multicolumn{" + str(5 + len(ES_QUALITY_COLS) + 4) + r"}{l}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r"\# & Model & Experiment & Think & SF & "
        + metric_heads
        + r" & sec & inp.tok & out.tok & tot.tok \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        "\\multicolumn{" + str(5 + len(ES_QUALITY_COLS) + 4) + r"}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

def es_footer() -> list[str]:
    return [
        r"\end{longtable}",
        r"\end{footnotesize}",
        r"\end{landscape}",
    ]

def es_data_row(row_label: str, row: dict, is_best_overall: bool,
                is_highlighted: bool) -> str:
    signed = row.get("sf") == "Δ"
    label = esc(row_label)
    if is_best_overall:
        label = r"\textrightarrow{} " + label
    model = esc(row["model"])
    exp   = esc(row["experiment"])
    think = esc(row.get("think", ""))
    sf    = esc(row.get("sf", ""))

    vals = []
    for m in ES_QUALITY_COLS:
        v = row.get(f"overall_{m}")
        vals.append(fmtv(v))
    vals.append(fmtv(row.get("mean_example_seconds")))
    vals.append(fmtv(row.get("mean_input_tokens")))
    vals.append(fmtv(row.get("mean_output_tokens")))
    vals.append(fmtv(row.get("mean_total_tokens")))

    cells = " & ".join([label, model, exp, think, sf] + vals)
    if is_highlighted:
        return r"\rowcolor{tabhighlight}" + cells + r" \\"
    return cells + r" \\"


def build_es_table(dev_slug: str) -> list[str]:
    titles = {
        "sns1064":     "Spanish SNS1064 dev ablation results (overall section)",
        "casimedicos": "Spanish CasiMedicos dev ablation results (overall section)",
        "mixed":       "Spanish SNS1064+CasiMedicos dev ablation results (overall section)",
    }
    label_map = {"sns1064": "tab:es-sns", "casimedicos": "tab:es-casi", "mixed": "tab:es-mixed"}

    nosf_start, sf_start = QWEN_DEV_IDS[dev_slug]
    all_rows: list[dict] = []
    for i, ((m_label, m_run), q_slug, q_label) in enumerate(
            zip(MISTRAL_RUNS[dev_slug], QWEN_SLUGS[dev_slug], QWEN_EXP_LABELS)):
        all_rows.extend(mistral_rows(i, m_label, m_run))
        all_rows.extend(qwen_rows(i, q_label, dev_slug, q_slug, nosf_start, sf_start, i))

    best_mistral = _best_key_for(all_rows, "Mistral-7B-Instruct")
    best_qwen    = _best_key_for(all_rows, "Qwen3.5-9B")
    best_overall = _best_key_for(all_rows)

    lines = es_header(titles[dev_slug], label_map[dev_slug])

    exp_model_seen: set[tuple] = set()
    exp_num = -1
    prev_exp_idx: int | None = None

    for row in all_rows:
        exp_idx = row["exp_idx"]
        model_suffix = "a" if row["model"] == "Mistral-7B-Instruct" else "b"
        if exp_idx != prev_exp_idx:
            exp_num += 1
            prev_exp_idx = exp_idx
            # separator between experiment groups
            if exp_num > 0:
                lines.append(r"\midrule")
        key = (exp_idx, model_suffix)
        if key not in exp_model_seen:
            row_label = f"{exp_num}{model_suffix}"
            exp_model_seen.add(key)
        else:
            row_label = ""

        rk = row["_key"]
        is_highlighted = rk in (best_mistral, best_qwen)
        is_best_overall = rk == best_overall
        lines.append(es_data_row(row_label, row, is_best_overall, is_highlighted))

    lines.extend(es_footer())
    return lines


# ── EU tables ─────────────────────────────────────────────────────────────────

EU_QUALITY_COLS = list(QUALITY_METRICS)
EU_COL_LABELS   = [QUALITY_METRIC_LABELS[m] for m in EU_QUALITY_COLS]

def eu_col_spec() -> str:
    return r"l l l l " + " ".join(["r"] * (len(EU_QUALITY_COLS) + 4))

def eu_header(caption: str, label: str) -> list[str]:
    metric_heads = " & ".join(esc(h) for h in EU_COL_LABELS)
    return [
        r"\begin{landscape}",
        r"\begin{footnotesize}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{eu_col_spec()}}}",
        rf"\caption{{{esc(caption)}}} \label{{{label}}} \\",
        r"\toprule",
        r"\# & Model & Experiment & SF & "
        + metric_heads
        + r" & sec & inp.tok & out.tok & tot.tok \\",
        r"\midrule",
        r"\endfirsthead",
        "\\multicolumn{" + str(4 + len(EU_QUALITY_COLS) + 4) + r"}{l}{\tablename\ \thetable{} -- continued} \\",
        r"\toprule",
        r"\# & Model & Experiment & SF & "
        + metric_heads
        + r" & sec & inp.tok & out.tok & tot.tok \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        "\\multicolumn{" + str(4 + len(EU_QUALITY_COLS) + 4) + r"}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

def eu_footer() -> list[str]:
    return [
        r"\end{longtable}",
        r"\end{footnotesize}",
        r"\end{landscape}",
    ]


def eu_data_row(row_label: str, r: dict, is_overall_best: bool, is_highlighted: bool) -> str:
    signed = r["signed"]
    label = esc(row_label)
    if is_overall_best:
        label = r"\textrightarrow{} " + label
    model = esc(r["model_label"])
    exp   = esc(r["experiment"])
    sf    = esc(r["self_feedback"])

    vals = []
    for m in EU_QUALITY_COLS:
        v = r.get(f"overall_{m}")
        vals.append(fmtv(v))
    vals.append(fmtv(r.get("mean_example_seconds")))
    vals.append(fmtv(r.get("mean_input_tokens")))
    vals.append(fmtv(r.get("mean_output_tokens")))
    vals.append(fmtv(r.get("mean_total_tokens")))

    cells = " & ".join([label, model, exp, sf] + vals)
    if is_highlighted:
        return r"\rowcolor{tabhighlight}" + cells + r" \\"
    return cells + r" \\"


def build_eu_table(dev_slug: str) -> list[str]:
    titles = {
        "sns1064_eu":     "Basque SNS1064 EU dev ablation results (overall section)",
        "casimedicos_eu": "Basque CasiMedicos EU dev ablation results (overall section)",
        "mixed_eu":       "Basque SNS1064+CasiMedicos EU dev ablation results (overall section)",
    }
    label_map = {
        "sns1064_eu":     "tab:eu-sns",
        "casimedicos_eu": "tab:eu-casi",
        "mixed_eu":       "tab:eu-mixed",
    }

    all_rows = eu_build_all_rows()
    dev_rows = [r for r in all_rows if r["dataset_slug"] == dev_slug]

    # sort: experiment order, then model, then SF condition
    def sort_key(r):
        e_idx = EXPERIMENT_ORDER.index(r["experiment_slug"]) if r["experiment_slug"] in EXPERIMENT_ORDER else 99
        m_idx = MODEL_ORDER.get(r["model_label"], 99)
        sf_idx = {"noSF": 0, "SF": 1, "Δ": 2}.get(r["self_feedback"], 9)
        return (e_idx, m_idx, sf_idx)
    dev_rows.sort(key=sort_key)

    # best per model (noSF rows only)
    best_keys: set[tuple] = set()
    for model_label in MODEL_ORDER:
        model_nosf = [r for r in dev_rows if r["model_label"] == model_label and r["self_feedback"] == "noSF"]
        if model_nosf:
            best = max(model_nosf, key=lambda r: float(r.get("overall_bertscore_f1") or 0))
            best_keys.add((best["model_label"], best["experiment_slug"], "noSF"))

    all_nosf = [r for r in dev_rows if r["self_feedback"] == "noSF" and r.get("overall_bertscore_f1") is not None]
    overall_best = max(all_nosf, key=lambda r: float(r.get("overall_bertscore_f1") or 0)) if all_nosf else None
    overall_best_key = (overall_best["model_label"], overall_best["experiment_slug"], "noSF") if overall_best else None

    lines = eu_header(titles[dev_slug], label_map[dev_slug])

    exp_num = -1
    prev_exp_slug: str | None = None
    for r in dev_rows:
        if r["self_feedback"] == "noSF":
            if r["experiment_slug"] != prev_exp_slug:
                exp_num += 1
                if exp_num > 0:
                    lines.append(r"\midrule")
                prev_exp_slug = r["experiment_slug"]
            suffix = "a" if r["model_label"] == "Llama-3.1-8B-Instruct" else "b"
            row_label = f"{exp_num}{suffix}"
        else:
            row_label = ""

        is_highlighted = (r["model_label"], r["experiment_slug"], r["self_feedback"]) in best_keys
        is_overall_best = overall_best_key is not None and (r["model_label"], r["experiment_slug"], r["self_feedback"]) == overall_best_key
        lines.append(eu_data_row(row_label, r, is_overall_best, is_highlighted))

    lines.extend(eu_footer())
    return lines


# ── write files ───────────────────────────────────────────────────────────────

def write_tex(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    for dev_slug in ("sns1064", "casimedicos", "mixed"):
        lines = build_es_table(dev_slug)
        write_tex(OUT_DIR / f"appendix_es_{dev_slug}.tex", lines)

    for dev_slug in ("sns1064_eu", "casimedicos_eu", "mixed_eu"):
        lines = build_eu_table(dev_slug)
        write_tex(OUT_DIR / f"appendix_eu_{dev_slug}.tex", lines)


if __name__ == "__main__":
    main()
