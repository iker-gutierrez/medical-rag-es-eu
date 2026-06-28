#!/usr/bin/env python
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports/metrics"
CONFIG_DIR = ROOT / "configs/experiments"
OUT_MD = METRICS_DIR / "eu_dev_ablation_results.md"
OUT_CSV = METRICS_DIR / "eu_dev_ablation_results.csv"

SECTIONS = ("short_answer", "evidence", "overall")
SECTION_LABELS = {"short_answer": "answer", "evidence": "evidence", "overall": "overall"}

QUALITY_METRICS = (
    "token_precision",
    "token_recall",
    "token_overlap_f1",
    "rouge_l_f1",
    "cosine_similarity",
    "bertscore_f1",
    "answer_context_token_f1",
    "gold_context_token_f1",
)
QUALITY_METRIC_LABELS = {
    "token_precision": "Token_prec",
    "token_recall": "Token_rec",
    "token_overlap_f1": "token F1",
    "rouge_l_f1": "ROUGE-L",
    "cosine_similarity": "Cosim",
    "bertscore_f1": "BERTScore",
    "answer_context_token_f1": "answer-context F1",
    "gold_context_token_f1": "gold-context F1",
}

DEV_SLUGS = ["sns1064_eu", "casimedicos_eu", "mixed_eu"]
DEV_LABELS = {
    "sns1064_eu": "SNS1064 EU",
    "casimedicos_eu": "CasiMedicos EU",
    "mixed_eu": "SNS1064+CasiMedicos EU",
}
DEV_SIZES = {
    "sns1064_eu": 106,
    "casimedicos_eu": 55,
    "mixed_eu": 161,
}

MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "HiTZ/Latxa-Llama-3.1-8B-Instruct": "Latxa-Llama-3.1-8B-Instruct",
}
MODEL_ORDER = {"Llama-3.1-8B-Instruct": 0, "Latxa-Llama-3.1-8B-Instruct": 1}

EXPERIMENT_LABELS = {
    "no_rag": "Baseline LLM only",
    "rag_e5_topk1": "e5 top 1",
    "rag_e5_topk3": "e5 top 3",
    "rag_e5_topk5": "e5 top 5",
    "rag_e5_rerank1": "rerank top 1",
    "rag_e5_rerank3": "rerank top 3",
    "rag_e5_rerank5": "rerank top 5",
    "3shot_no_rag": "3-shot, no RAG",
    "rag_3shot_e5_rerank5": "3-shot + rerank top 5",
    "rag_cross_domain_e5_rerank5": "cross-domain retrieval",
    "rag_mixed_e5_rerank5": "mixed-domain retrieval",
    "rag_sns1064_e5_rerank5": "SNS-only retrieval",
    "rag_casimedicos_e5_rerank5": "CasiMedicos-only retrieval",
}
EXPERIMENT_ORDER = [
    "no_rag",
    "rag_e5_topk1",
    "rag_e5_topk3",
    "rag_e5_topk5",
    "rag_e5_rerank1",
    "rag_e5_rerank3",
    "rag_e5_rerank5",
    "3shot_no_rag",
    "rag_3shot_e5_rerank5",
    "rag_cross_domain_e5_rerank5",
    "rag_mixed_e5_rerank5",
    "rag_sns1064_e5_rerank5",
    "rag_casimedicos_e5_rerank5",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def infer_dataset_slug(name: str) -> str:
    for slug in sorted(DEV_LABELS, key=len, reverse=True):
        if name.endswith(f"_{slug}"):
            return slug
    return ""


def infer_model_key(name: str) -> str:
    if name.startswith("latxa_llama31_8b_"):
        return "latxa_llama31_8b"
    if name.startswith("llama31_8b_"):
        return "llama31_8b"
    return ""


def infer_experiment_slug(name: str) -> str:
    ds = infer_dataset_slug(name)
    mk = infer_model_key(name)
    prefix = f"{mk}_"
    suffix = f"_extractive_{ds}"
    if not mk or not ds:
        return name
    return name[len(prefix): -len(suffix)]


def load_config(path: Path) -> dict[str, Any] | None:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    exp_name = cfg.get("experiment_name", "")
    ds = infer_dataset_slug(exp_name)
    mk = infer_model_key(exp_name)
    if not ds or not mk:
        return None
    exp_slug = infer_experiment_slug(exp_name)
    model_id = cfg.get("model", "")
    return {
        "run_id": path.stem,
        "model_id": model_id,
        "model_label": MODEL_LABELS.get(model_id, model_id),
        "dataset_slug": ds,
        "dev_set": DEV_LABELS[ds],
        "experiment_slug": exp_slug,
        "experiment": EXPERIMENT_LABELS.get(exp_slug, exp_slug),
    }


def load_summary(run_id: str) -> dict[str, Any] | None:
    path = METRICS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("summary")


def nested_get(data: dict | None, keys: list[str]) -> Any:
    v: Any = data
    for k in keys:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None or value == "":
        return ""
    n = float(value)
    text = f"{n:.2f}"
    if signed and n > 0:
        return "+" + text
    return text


# ── cost helpers (same logic as write_supervisor_summary.py) ─────────────────

def _mean(d: dict, key: str) -> float:
    return float((d.get(key) or {}).get("mean") or 0.0)


def no_sf_cost(summary: dict | None) -> dict[str, float | None]:
    if not summary:
        return dict.fromkeys(["mean_example_seconds", "mean_input_tokens", "mean_output_tokens", "mean_total_tokens"])
    cost = summary.get("cost", {})
    timing = cost.get("timing", {})
    tokens = cost.get("token_counts", {})
    seconds = sum(_mean(timing, k) for k in [
        "retrieval_seconds", "rerank_seconds", "few_shot_seconds",
        "prompt_seconds", "generation_seconds",
    ])
    inp = _mean(tokens, "input_tokens")
    out = _mean(tokens, "initial_output_tokens")
    return {
        "mean_example_seconds": seconds,
        "mean_input_tokens": inp,
        "mean_output_tokens": out,
        "mean_total_tokens": inp + out,
    }


def sf_cost(summary: dict | None) -> dict[str, float | None]:
    if not summary:
        return dict.fromkeys(["mean_example_seconds", "mean_input_tokens", "mean_output_tokens", "mean_total_tokens"])
    cost = summary.get("cost", {})
    timing = cost.get("timing", {})
    tokens = cost.get("token_counts", {})
    inp = _mean(tokens, "input_tokens") + _mean(tokens, "feedback_input_tokens")
    out = _mean(tokens, "output_tokens")
    total = _mean(tokens, "total_tokens") or inp + out
    return {
        "mean_example_seconds": _mean(timing, "example_seconds"),
        "mean_input_tokens": inp,
        "mean_output_tokens": out,
        "mean_total_tokens": total,
    }


def delta_cost(summary: dict | None) -> dict[str, float | None]:
    if not summary:
        return dict.fromkeys(["mean_example_seconds", "mean_input_tokens", "mean_output_tokens", "mean_total_tokens"])
    b = no_sf_cost(summary)
    a = sf_cost(summary)
    return {k: None if b[k] is None or a[k] is None else a[k] - b[k] for k in b}


def section_metrics(summary: dict | None, condition: str, section: str) -> dict[str, Any]:
    if not summary:
        return {}
    if condition == "noSF":
        return (summary.get("before_feedback") or summary).get(section, {})
    if condition == "SF":
        return summary.get("after_feedback", {}).get(section, {})
    if condition == "Δ":
        return summary.get("self_feedback_delta", {}).get(section, {})
    return {}


def cost_for(summary: dict | None, condition: str) -> dict[str, float | None]:
    if condition == "noSF":
        return no_sf_cost(summary)
    if condition == "SF":
        return sf_cost(summary)
    return delta_cost(summary)


# ── HTML helpers ──────────────────────────────────────────────────────────────

def cell_style(*, strong_left: bool = False) -> str:
    s = "border-left: 1px solid #d0d7de; border-right: 1px solid #d0d7de; padding: 4px 6px;"
    if strong_left:
        s += " border-left: 2px solid #57606a;"
    return s


def th(content: str, *, colspan: int | None = None, rowspan: int | None = None, strong_left: bool = False) -> str:
    attrs = [f'style="{cell_style(strong_left=strong_left)}"']
    if colspan is not None:
        attrs.append(f'colspan="{colspan}"')
    if rowspan is not None:
        attrs.append(f'rowspan="{rowspan}"')
    return f"<th {' '.join(attrs)}>{html.escape(content)}</th>"


def td(content: Any, *, strong_left: bool = False) -> str:
    return f'<td style="{cell_style(strong_left=strong_left)}">{html.escape(str(content))}</td>'


# ── row building ──────────────────────────────────────────────────────────────

def build_all_rows() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for cfg_path in sorted(CONFIG_DIR.glob("1*_*.json")):
        run_num = int(cfg_path.name.split("_", 1)[0])
        if run_num < 106 or run_num > 171:
            continue
        spec = load_config(cfg_path)
        if spec is None:
            continue
        specs.append(spec)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        summary = load_summary(spec["run_id"])
        for condition in ("noSF", "SF", "Δ"):
            signed = condition == "Δ"
            cost = cost_for(summary, condition)
            row: dict[str, Any] = {
                **spec,
                "self_feedback": condition,
                "signed": signed,
                "status": "done" if summary is not None else "missing",
                **cost,
            }
            for section in SECTIONS:
                sm = section_metrics(summary, condition, section)
                for metric in QUALITY_METRICS:
                    row[f"{section}_{metric}"] = sm.get(metric)
            rows.append(row)

    def sort_key(r: dict) -> tuple:
        ds_idx = DEV_SLUGS.index(r["dataset_slug"]) if r["dataset_slug"] in DEV_SLUGS else 99
        e_idx = EXPERIMENT_ORDER.index(r["experiment_slug"]) if r["experiment_slug"] in EXPERIMENT_ORDER else 99
        m_idx = MODEL_ORDER.get(r["model_label"], 99)
        sf_idx = {"noSF": 0, "SF": 1, "Δ": 2}.get(r["self_feedback"], 9)
        return (ds_idx, e_idx, m_idx, sf_idx)

    return sorted(rows, key=sort_key)


# ── HTML table for one dev set ────────────────────────────────────────────────

def build_html_table(rows: list[dict[str, Any]], dev_slug: str) -> str:
    dev_rows = [r for r in rows if r["dataset_slug"] == dev_slug]
    quality_span = len(QUALITY_METRICS) * len(SECTIONS)

    header_1 = (
        th("#", rowspan=3)
        + th("model", rowspan=3)
        + th("experiment", rowspan=3)
        + th("self-feedback", rowspan=3)
        + th("Quality ↑", colspan=quality_span, strong_left=True)
        + th("Cost ↓", colspan=4, strong_left=True)
    )
    header_2 = "".join(
        th(QUALITY_METRIC_LABELS[m], colspan=len(SECTIONS), strong_left=True)
        for m in QUALITY_METRICS
    )
    header_2 += th("sec/sample", rowspan=2, strong_left=True)
    header_2 += th("tokens/sample", colspan=3, strong_left=True)
    header_3 = ""
    for _m in QUALITY_METRICS:
        for idx, sec in enumerate(SECTIONS):
            header_3 += th(SECTION_LABELS[sec], strong_left=idx == 0)
    header_3 += th("input", strong_left=True) + th("output") + th("total")

    # best noSF row (highest overall bertscore) per model
    best_keys: set[tuple[str, str, str]] = set()
    for model_label in MODEL_ORDER:
        model_nosf = [r for r in dev_rows if r["model_label"] == model_label and r["self_feedback"] == "noSF"]
        if model_nosf:
            best = max(model_nosf, key=lambda r: float(r.get("overall_bertscore_f1") or 0))
            best_keys.add((best["model_label"], best["experiment_slug"], "noSF"))

    # best row across all models (☞ marker)
    all_nosf = [r for r in dev_rows if r["self_feedback"] == "noSF" and r.get("overall_bertscore_f1") is not None]
    overall_best = max(all_nosf, key=lambda r: float(r.get("overall_bertscore_f1") or 0)) if all_nosf else None
    overall_best_key = (overall_best["model_label"], overall_best["experiment_slug"], "noSF") if overall_best else None

    table_rows: list[str] = []
    exp_num = -1
    prev_exp_slug: str | None = None
    for r in dev_rows:
        signed = r["signed"]
        is_best = (r["model_label"], r["experiment_slug"], r["self_feedback"]) in best_keys
        is_overall_best = overall_best_key is not None and (r["model_label"], r["experiment_slug"], r["self_feedback"]) == overall_best_key
        if r["self_feedback"] == "noSF":
            if r["experiment_slug"] != prev_exp_slug:
                exp_num += 1
                prev_exp_slug = r["experiment_slug"]
            suffix = "a" if r["model_label"] == "Llama-3.1-8B-Instruct" else "b"
            row_label = f"{exp_num}{suffix}"
        else:
            row_label = ""
        if is_overall_best:
            row_label = f"☞ {row_label}" if row_label else "☞"
        cells = [
            td(row_label),
            td(r["model_label"]),
            td(r["experiment"]),
            td(r["self_feedback"]),
        ]
        for metric in QUALITY_METRICS:
            for idx, section in enumerate(SECTIONS):
                cells.append(td(fmt(r.get(f"{section}_{metric}"), signed=signed), strong_left=idx == 0))
        cells.extend([
            td(fmt(r.get("mean_example_seconds"), signed=signed), strong_left=True),
            td(fmt(r.get("mean_input_tokens"), signed=signed), strong_left=True),
            td(fmt(r.get("mean_output_tokens"), signed=signed)),
            td(fmt(r.get("mean_total_tokens"), signed=signed)),
        ])
        row_style = ' style="background-color: #fff8c5; font-weight: 700;"' if is_best else ""
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


# ── summary table for one dev set ─────────────────────────────────────────────

def build_summary_table(rows: list[dict[str, Any]], dev_slug: str) -> str:
    dev_rows = [r for r in rows if r["dataset_slug"] == dev_slug and r["self_feedback"] == "noSF"]
    lines = [
        "| # | Model | Experiment | BERTScore | SF Δ |",
        "|---:|---|---|---:|---:|",
    ]
    exp_num = -1
    prev_exp: str | None = None
    for r in dev_rows:
        if r["experiment_slug"] != prev_exp:
            exp_num += 1
            prev_exp = r["experiment_slug"]
        suffix = "a" if r["model_label"] == "Llama-3.1-8B-Instruct" else "b"
        label = f"{exp_num}{suffix}"
        bert = fmt(r.get("overall_bertscore_f1"))
        delta_rows = [x for x in rows
                      if x["dataset_slug"] == dev_slug
                      and x["model_label"] == r["model_label"]
                      and x["experiment_slug"] == r["experiment_slug"]
                      and x["self_feedback"] == "Δ"]
        delta = fmt(delta_rows[0].get("overall_bertscore_f1"), signed=True) if delta_rows else ""
        lines.append(f"| {label} | {r['model_label']} | {r['experiment']} | {bert} | {delta} |")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dev_set", "model_label", "experiment", "self_feedback", "run_id", "status",
        "mean_example_seconds", "mean_input_tokens", "mean_output_tokens", "mean_total_tokens",
    ]
    for section in SECTIONS:
        for metric in QUALITY_METRICS:
            fieldnames.append(f"{section}_{metric}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_all_rows()

    sections: list[str] = [
        "# EU dev ablation summary",
        "",
        "Supervisor-facing summary for the Basque dev ablations. Compares"
        " `meta-llama/Llama-3.1-8B-Instruct` (parent model) with"
        " `HiTZ/Latxa-Llama-3.1-8B-Instruct` (Basque-adapted) across the full ablation grid.",
        "",
        "## Experimental setup",
        "",
        "- Dev sets: SNS1064 EU, CasiMedicos EU, and SNS1064+CasiMedicos EU.",
        "- Generators: Llama-3.1-8B-Instruct and Latxa-Llama-3.1-8B-Instruct.",
        "- Main prompt family: cleaned extractive prompt (Basque).",
        "- Main retrieval stack: multilingual-e5 FAISS index + multilingual MS MARCO cross-encoder reranker.",
        "- Main reported row: no self-feedback. Self-feedback is tracked within each run.",
        "",
        "## Current results",
        "",
    ]

    for slug in DEV_SLUGS:
        label = DEV_LABELS[slug]
        n = DEV_SIZES[slug]
        sections += [
            f"### {label} dev results",
            "",
            f"{label} dev set, {n} examples, open-answer task.",
            "",
            "All quality scores are out of 100. Cost is experiment-level and is reported per sample.",
            "",
            build_html_table(rows, slug),
            "",
            "#### Takeaways",
            "",
            f"**Core findings ({label} dev set, no self-feedback):**",
            "",
            build_summary_table(rows, slug),
            "",
        ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(sections), encoding="utf-8")
    write_csv(rows)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
