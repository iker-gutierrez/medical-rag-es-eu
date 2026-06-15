#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "metrics"
OUTPUT = METRICS_DIR / "agentic_dev_results.md"

ROWS = [
    (
        "SNS1064 dev",
        "Baseline LLM only",
        "Best non-baseline: exp 8 (3-shot + rerank top5)",
        "Agentic: baseline vs exp 8",
        "17_mistral7b_no_rag_no_think_extractive_sf_dev",
        "38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev",
        "61_mistral7b_agentic_baseline_vs_exp8_sns1064_dev",
    ),
    (
        "CasiMedicos dev",
        "Baseline LLM only",
        "Best non-baseline: exp 2 (e5 top3)",
        "Agentic: baseline vs exp 2",
        "39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev",
        "41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev",
        "62_mistral7b_agentic_baseline_vs_exp2_casimedicos_dev",
    ),
    (
        "SNS1064+CasiMedicos dev",
        "Baseline LLM only",
        "Best non-baseline: exp 8 (3-shot + rerank top5)",
        "Agentic: baseline vs exp 8",
        "50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev",
        "58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev",
        "63_mistral7b_agentic_baseline_vs_exp8_mixed_dev",
    ),
]

SECTIONS = ("short_answer", "evidence", "overall")
SECTION_LABELS = {
    "short_answer": "answer",
    "evidence": "evidence",
    "overall": "overall",
}


def load_summary(run_id: str) -> dict[str, Any]:
    return json.loads((METRICS_DIR / f"{run_id}.json").read_text(encoding="utf-8"))["summary"]


def no_sf_block(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("before_feedback") or summary


def metric(summary: dict[str, Any], section: str, name: str) -> float:
    return float(no_sf_block(summary)[section][name])


def mean_seconds(summary: dict[str, Any]) -> float:
    return float(summary["cost"]["timing"]["example_seconds"]["mean"])


def mean_tokens(summary: dict[str, Any]) -> float:
    return float(summary["cost"]["token_counts"]["total_tokens"]["mean"])


def fmt(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def main() -> None:
    lines = [
        "# Agentic dev results",
        "",
        (
            "Dev-only agentic reasoner experiment. The judge receives the no-self-feedback "
            "baseline answer and the no-self-feedback answer from the best non-baseline "
            "configuration for that dev set. No test-set evaluation is included."
        ),
        "",
        "| dev set | agentic BERT | Δ BERT vs best | agentic Cosim | Δ Cosim vs best | agentic sec/sample | Δ sec vs best | agentic tokens/sample | Δ tokens vs best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for dev_name, _baseline_label, _candidate_label, _agentic_label, baseline_run, candidate_run, agentic_run in ROWS:
        baseline = load_summary(baseline_run)
        candidate = load_summary(candidate_run)
        agentic = load_summary(agentic_run)
        candidate_bert = metric(candidate, "overall", "bertscore_f1")
        agentic_bert = metric(agentic, "overall", "bertscore_f1")
        candidate_cosim = metric(candidate, "overall", "cosine_similarity")
        agentic_cosim = metric(agentic, "overall", "cosine_similarity")
        candidate_seconds = mean_seconds(candidate)
        agentic_seconds = mean_seconds(agentic)
        candidate_tokens = mean_tokens(candidate)
        agentic_tokens = mean_tokens(agentic)
        lines.append(
            "| "
            + " | ".join(
                [
                    dev_name,
                    fmt(agentic_bert),
                    signed(agentic_bert - candidate_bert),
                    fmt(agentic_cosim),
                    signed(agentic_cosim - candidate_cosim),
                    fmt(agentic_seconds),
                    signed(agentic_seconds - candidate_seconds),
                    f"{agentic_tokens:.0f}",
                    f"{agentic_tokens - candidate_tokens:+.0f}",
                ]
            )
            + " |"
        )

    for dev_name, baseline_label, candidate_label, agentic_label, baseline_run, candidate_run, agentic_run in ROWS:
        lines.extend(
            [
                "",
                f"## {dev_name}",
                "",
                "| system | section | Cosim | BERTScore | token F1 | ROUGE-L | sec/sample | tokens/sample |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, run_id in (
            (baseline_label, baseline_run),
            (candidate_label, candidate_run),
            (agentic_label, agentic_run),
        ):
            summary = load_summary(run_id)
            for section in SECTIONS:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            label,
                            SECTION_LABELS[section],
                            fmt(metric(summary, section, "cosine_similarity")),
                            fmt(metric(summary, section, "bertscore_f1")),
                            fmt(metric(summary, section, "token_overlap_f1")),
                            fmt(metric(summary, section, "rouge_l_f1")),
                            fmt(mean_seconds(summary)),
                            f"{mean_tokens(summary):.0f}",
                        ]
                    )
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            (
                "- The agentic reasoner did not improve over the selected best non-baseline "
                "configuration on overall BERTScore in any dev set."
            ),
            (
                "- The best non-baseline system remains the stronger choice so far: exp 8 "
                "for SNS1064, exp 2 for CasiMedicos, and exp 8 for the mixed dev set."
            ),
            (
                "- The agentic reasoner is substantially more expensive because its cost "
                "includes generating the baseline answer, generating the best-system answer, "
                "and then running the judge/verifier call."
            ),
            "",
        ]
    )

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
