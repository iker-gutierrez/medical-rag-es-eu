#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mixed_agentic_tasks import METRICS_DIR, ROOT, TASKS


OUTPUT = METRICS_DIR / "mixed_agentic_dev_results.md"
SECTIONS = ("short_answer", "evidence", "overall")
SECTION_LABELS = {"short_answer": "answer", "evidence": "evidence", "overall": "overall"}


def load_summary(run_id: str) -> dict[str, Any] | None:
    path = METRICS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def no_sf(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("before_feedback") or summary


def metric(summary: dict[str, Any], section: str, name: str) -> float:
    return float(no_sf(summary)[section][name])


def mean_seconds(summary: dict[str, Any]) -> float:
    return float(summary["cost"]["timing"]["example_seconds"]["mean"])


def mean_tokens(summary: dict[str, Any]) -> float:
    return float(summary["cost"]["token_counts"]["total_tokens"]["mean"])


def fmt(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def metadata(run_id: str) -> dict[str, Any]:
    path = ROOT / "experiments" / "runs" / run_id / "predictions.meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    lines = [
        "# Mixed agentic dev results",
        "",
        (
            "Additional dev-only mixed Agentic Reasoner experiments. These do not overwrite "
            "the original unmixed Agentic Reasoner runs."
        ),
        "",
        "| task | dev set | baseline source | selected candidate | judge | BERT | Δ BERT vs candidate | Cosim | Δ Cosim vs candidate | sec/sample | tokens/sample |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for idx, task in enumerate(TASKS):
        agentic = load_summary(task.run_id)
        meta = metadata(task.run_id)
        baseline_run = str(meta.get("baseline_predictions", "")).split("/")[-2] if meta else "pending"
        candidate_run = str(meta.get("candidate_predictions", "")).split("/")[-2] if meta else "pending"
        candidate = load_summary(candidate_run) if candidate_run != "pending" else None
        if agentic and candidate:
            agentic_bert = metric(agentic, "overall", "bertscore_f1")
            candidate_bert = metric(candidate, "overall", "bertscore_f1")
            agentic_cosim = metric(agentic, "overall", "cosine_similarity")
            candidate_cosim = metric(candidate, "overall", "cosine_similarity")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        task.dev_set,
                        baseline_run,
                        candidate_run,
                        task.judge_model_label,
                        fmt(agentic_bert),
                        signed(agentic_bert - candidate_bert),
                        fmt(agentic_cosim),
                        signed(agentic_cosim - candidate_cosim),
                        fmt(mean_seconds(agentic)),
                        f"{mean_tokens(agentic):.0f}",
                    ]
                )
                + " |"
            )
        else:
            lines.append(
                f"| {idx} | {task.dev_set} | {baseline_run} | {candidate_run} | {task.judge_model_label} | pending | pending | pending | pending | pending | pending |"
            )

    for idx, task in enumerate(TASKS):
        agentic = load_summary(task.run_id)
        if not agentic:
            continue
        lines.extend(
            [
                "",
                f"## Task {idx}: {task.dev_set}, {task.baseline_model} + {task.judge_model_label}",
                "",
                "| section | Cosim | BERTScore | token F1 | ROUGE-L | sec/sample | tokens/sample |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for section in SECTIONS:
            lines.append(
                "| "
                + " | ".join(
                    [
                        SECTION_LABELS[section],
                        fmt(metric(agentic, section, "cosine_similarity")),
                        fmt(metric(agentic, section, "bertscore_f1")),
                        fmt(metric(agentic, section, "token_overlap_f1")),
                        fmt(metric(agentic, section, "rouge_l_f1")),
                        fmt(mean_seconds(agentic)),
                        f"{mean_tokens(agentic):.0f}",
                    ]
                )
                + " |"
            )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
