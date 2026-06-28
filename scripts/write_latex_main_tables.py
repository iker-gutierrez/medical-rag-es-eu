"""Generate compact LaTeX longtable files for the main text.

Metrics shown (overall section only): ROUGE-L, Cosim, BERT-F1, sec/sample, tok/sample.
Uses footnotesize + longtable so tables span pages cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from update_es_ablation_with_qwen3 import (
    MISTRAL_RUNS, QWEN_EXP_LABELS, QWEN_SLUGS, QWEN_DEV_IDS,
    mistral_rows, qwen_rows, _best_key_for,
)
from write_eu_ablation_summary import (
    MODEL_ORDER, EXPERIMENT_ORDER,
    build_all_rows as eu_build_all_rows,
)

OUT_DIR = REPO / "manuscript"

# Quality metrics to show + cost
MAIN_METRICS  = ("rouge_l_f1", "cosine_similarity", "bertscore_f1")
METRIC_LABELS = {"rouge_l_f1": "ROUGE-L", "cosine_similarity": "Cosim", "bertscore_f1": "BERT-F1"}


def esc(s: str) -> str:
    return (str(s)
            .replace("&",  r"\&")
            .replace("%",  r"\%")
            .replace("_",  r"\_")
            .replace("#",  r"\#")
            .replace("☞",  r"$\Rightarrow$")
            .replace("Δ",  r"$\Delta$")
            .replace("~",  r"\textasciitilde{}"))


def fmtv(v, integer: bool = False) -> str:
    if v is None:
        return "---"
    try:
        f = float(v)
        return f"{f:.0f}" if integer else f"{f:.2f}"
    except (ValueError, TypeError):
        return str(v)


# ── ES table ───────────────────────────────────────────────────────────────────
# Columns: #(l) Model(l) Experiment(l) Think(l) SF(l) ROUGE-L(r) Cosim(r) BERT-F1(r) sec(r) tok(r)
ES_COL  = r"l l l l l r r r r r"
ES_NCOL = 10
ES_HEAD = (r"\# & Model & Experiment & Think & SF & "
           r"\multicolumn{1}{r}{ROUGE-L} & \multicolumn{1}{r}{Cosim} & "
           r"\multicolumn{1}{r}{BERT-F1} & \multicolumn{1}{r}{sec} & \multicolumn{1}{r}{tok} \\")


def es_lt_header(caption: str, label: str) -> list[str]:
    cont = "\\multicolumn{" + str(ES_NCOL) + r"}{l}{\tablename\ \thetable{} -- continued} \\"
    cont_foot = "\\multicolumn{" + str(ES_NCOL) + r"}{r}{Continued on next page} \\"
    return [
        r"\begin{footnotesize}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{ES_COL}}}",
        rf"\caption{{{esc(caption)}}} \label{{{label}}} \\",
        r"\toprule",
        ES_HEAD,
        r"\midrule",
        r"\endfirsthead",
        cont,
        r"\toprule",
        ES_HEAD,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        cont_foot,
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]


def es_lt_footer() -> list[str]:
    return [r"\end{longtable}", r"\end{footnotesize}"]


def es_data_row(row_label: str, row: dict, is_best_overall: bool, is_highlighted: bool) -> str:
    label = esc(row_label)
    if is_best_overall:
        label = r"$\Rightarrow$~" + label
    vals  = [fmtv(row.get(f"overall_{m}")) for m in MAIN_METRICS]
    vals += [fmtv(row.get("mean_example_seconds")),
             fmtv(row.get("mean_total_tokens"), integer=True)]
    cells = " & ".join([label,
                        esc(row["model"]),
                        esc(row["experiment"]),
                        esc(row.get("think", "")),
                        esc(row.get("sf", ""))] + vals)
    prefix = r"\rowcolor{tabhighlight}" if is_highlighted else ""
    return prefix + cells + r" \\"


def build_es_table(dev_slug: str) -> list[str]:
    captions = {
        "sns1064":
            "Spanish SNS1064 dev ablation (106 examples, open-answer). "
            "Quality on overall section: ROUGE-L, cosine similarity (Cosim), BERTScore F1 (BERT-F1). "
            "Cost: seconds per sample (sec), total tokens per sample (tok). "
            "Yellow = best per model; $\\Rightarrow$ = best overall.",
        "casimedicos":
            "Spanish CasiMedicos dev ablation (55 examples, multiple-choice). "
            "Same metrics as Table~\\ref{tab:es-main-sns}.",
        "mixed":
            "Spanish SNS1064+CasiMedicos dev ablation (161 examples, mixed). "
            "Same metrics as Table~\\ref{tab:es-main-sns}.",
    }
    labels = {"sns1064": "tab:es-main-sns", "casimedicos": "tab:es-main-casi", "mixed": "tab:es-main-mixed"}

    nosf_start, sf_start = QWEN_DEV_IDS[dev_slug]
    all_rows: list[dict] = []
    for i, ((m_label, m_run), q_slug, q_label) in enumerate(
            zip(MISTRAL_RUNS[dev_slug], QWEN_SLUGS[dev_slug], QWEN_EXP_LABELS)):
        all_rows.extend(mistral_rows(i, m_label, m_run))
        all_rows.extend(qwen_rows(i, q_label, dev_slug, q_slug, nosf_start, sf_start, i))

    best_mistral = _best_key_for(all_rows, "Mistral-7B-Instruct")
    best_qwen    = _best_key_for(all_rows, "Qwen3.5-9B")
    best_overall = _best_key_for(all_rows)

    lines = es_lt_header(captions[dev_slug], labels[dev_slug])

    exp_model_seen: set[tuple] = set()
    exp_num = -1
    prev_exp_idx: int | None = None

    for row in all_rows:
        exp_idx = row["exp_idx"]
        model_suffix = "a" if row["model"] == "Mistral-7B-Instruct" else "b"
        if exp_idx != prev_exp_idx:
            exp_num += 1
            if exp_num > 0:
                lines.append(r"\midrule")
            prev_exp_idx = exp_idx
        key = (exp_idx, model_suffix)
        if key not in exp_model_seen:
            row_label = f"{exp_num}{model_suffix}"
            exp_model_seen.add(key)
        else:
            row_label = ""

        rk = row["_key"]
        lines.append(es_data_row(
            row_label, row,
            is_best_overall=rk == best_overall,
            is_highlighted=rk in (best_mistral, best_qwen),
        ))

    lines.extend(es_lt_footer())
    return lines


# ── EU table ───────────────────────────────────────────────────────────────────
# Columns: #(l) Model(l) Experiment(l) SF(l) ROUGE-L(r) Cosim(r) BERT-F1(r) sec(r) tok(r)
EU_COL  = r"l l l l r r r r r"
EU_NCOL = 9
EU_HEAD = (r"\# & Model & Experiment & SF & "
           r"\multicolumn{1}{r}{ROUGE-L} & \multicolumn{1}{r}{Cosim} & "
           r"\multicolumn{1}{r}{BERT-F1} & \multicolumn{1}{r}{sec} & \multicolumn{1}{r}{tok} \\")


def eu_lt_header(caption: str, label: str) -> list[str]:
    cont = "\\multicolumn{" + str(EU_NCOL) + r"}{l}{\tablename\ \thetable{} -- continued} \\"
    cont_foot = "\\multicolumn{" + str(EU_NCOL) + r"}{r}{Continued on next page} \\"
    return [
        r"\begin{footnotesize}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{EU_COL}}}",
        rf"\caption{{{esc(caption)}}} \label{{{label}}} \\",
        r"\toprule",
        EU_HEAD,
        r"\midrule",
        r"\endfirsthead",
        cont,
        r"\toprule",
        EU_HEAD,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        cont_foot,
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]


def eu_lt_footer() -> list[str]:
    return [r"\end{longtable}", r"\end{footnotesize}"]


def eu_data_row(row_label: str, r: dict, is_best_overall: bool, is_highlighted: bool) -> str:
    label = esc(row_label)
    if is_best_overall:
        label = r"$\Rightarrow$~" + label
    vals  = [fmtv(r.get(f"overall_{m}")) for m in MAIN_METRICS]
    vals += [fmtv(r.get("mean_example_seconds")),
             fmtv(r.get("mean_total_tokens"), integer=True)]
    cells = " & ".join([label,
                        esc(r["model_label"]),
                        esc(r["experiment"]),
                        esc(r["self_feedback"])] + vals)
    prefix = r"\rowcolor{tabhighlight}" if is_highlighted else ""
    return prefix + cells + r" \\"


def build_eu_table(dev_slug: str) -> list[str]:
    captions = {
        "sns1064_eu":
            "Basque SNS1064 EU dev ablation (106 examples, open-answer). "
            "Same metrics as Table~\\ref{tab:es-main-sns}.",
        "casimedicos_eu":
            "Basque CasiMedicos EU dev ablation (55 examples, multiple-choice). "
            "Same metrics as Table~\\ref{tab:es-main-sns}.",
        "mixed_eu":
            "Basque SNS1064+CasiMedicos EU dev ablation (161 examples, mixed). "
            "Same metrics as Table~\\ref{tab:es-main-sns}.",
    }
    labels = {
        "sns1064_eu":     "tab:eu-main-sns",
        "casimedicos_eu": "tab:eu-main-casi",
        "mixed_eu":       "tab:eu-main-mixed",
    }

    all_rows = eu_build_all_rows()
    dev_rows = [r for r in all_rows if r["dataset_slug"] == dev_slug]

    def sort_key(r):
        e_idx = EXPERIMENT_ORDER.index(r["experiment_slug"]) if r["experiment_slug"] in EXPERIMENT_ORDER else 99
        m_idx = MODEL_ORDER.get(r["model_label"], 99)
        sf_idx = {"noSF": 0, "SF": 1, "Δ": 2}.get(r["self_feedback"], 9)
        return (e_idx, m_idx, sf_idx)
    dev_rows.sort(key=sort_key)

    best_keys: set[tuple] = set()
    for model_label in MODEL_ORDER:
        model_nosf = [r for r in dev_rows if r["model_label"] == model_label and r["self_feedback"] == "noSF"]
        if model_nosf:
            best = max(model_nosf, key=lambda r: float(r.get("overall_bertscore_f1") or 0))
            best_keys.add((best["model_label"], best["experiment_slug"], "noSF"))

    all_nosf = [r for r in dev_rows if r["self_feedback"] == "noSF" and r.get("overall_bertscore_f1") is not None]
    overall_best = max(all_nosf, key=lambda r: float(r.get("overall_bertscore_f1") or 0)) if all_nosf else None
    overall_best_key = (overall_best["model_label"], overall_best["experiment_slug"], "noSF") if overall_best else None

    lines = eu_lt_header(captions[dev_slug], labels[dev_slug])

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
        is_best_overall = overall_best_key is not None and \
            (r["model_label"], r["experiment_slug"], r["self_feedback"]) == overall_best_key
        lines.append(eu_data_row(row_label, r, is_best_overall, is_highlighted))

    lines.extend(eu_lt_footer())
    return lines


# ── write ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for dev_slug in ("sns1064", "casimedicos", "mixed"):
        path = OUT_DIR / f"main_table_es_{dev_slug}.tex"
        path.write_text("\n".join(build_es_table(dev_slug)) + "\n")
        print(f"Wrote {path}")

    for dev_slug in ("sns1064_eu", "casimedicos_eu", "mixed_eu"):
        path = OUT_DIR / f"main_table_eu_{dev_slug}.tex"
        path.write_text("\n".join(build_eu_table(dev_slug)) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
