#!/usr/bin/env python
"""Smoke test for the staged-eval approach (scripts/staged_ablation_runner.py):
proves that infer -> evaluate -> decide -> infer next stage actually works end
to end on real Slurm/vLLM infrastructure, not just in a --dry-run print.

Runs the exact same orchestration functions staged_ablation_runner.py uses
(submit_infer_array, wait_for_job, evaluate, best_by_meanq, base_retrieval_fields,
apply_base) against a throwaway copy of Latxa's 11-config spec -- chosen because
it is the fastest model in the ablation (~0.2-0.5s/sample at seed42 timing) -- with
every config's --limit cut to 4 records and self-feedback off, so the whole thing
finishes in minutes instead of the hours a real staged run takes, while still
being genuine GPU inference through the real vLLM path, not a mock.

It does NOT touch any real experiment ID (1051-1061) or real config file: every
smoke config is a copy under a "smoke_stagedeval_" name, output to
experiments/runs/smoke_stagedeval_*, evaluated to reports/metrics/smoke_stagedeval_*.
Real production predictions and configs are never read or written.

What it proves, concretely:
  1. Stage A (baseline + 6-config retrieval sweep) runs, evaluates, and MeanQ
     picks a winner from real (if tiny) generated text -- not canned data.
  2. Row 8's config file is REWRITTEN with the winner's retrieval fields BEFORE
     stage B is submitted (the central claim of the staged approach: no row is
     ever inferred with a base that gets discarded and re-run).
  3. Stage B runs on the corrected config, evaluates, and MeanQ over rows 1-6+8
     produces a second decision.
  4. Domain rows (9-10) are wired to that decision and stage C runs to completion.

Usage: python scripts/smoke_test_staged_eval.py
Cleans up its own smoke_stagedeval_* run/metric/config files on start (so it is
safe to re-run) but leaves them in place after a successful run as evidence.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from staged_ablation_runner import submit_infer_array, wait_for_job, evaluate  # noqa: E402
from rewire_dependent_configs import base_retrieval_fields, apply_base  # noqa: E402
from meanq import best_by_meanq  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "experiments"
RUNS = ROOT / "experiments" / "runs"
METRICS = ROOT / "reports" / "metrics"
SMOKE_TAG = "smoke_stagedeval"
LIMIT = 4  # records per config: enough to exercise both source types

# (label, real id_prefix, real base) for Latxa's 11 real configs, reused as the
# template this smoke test copies from -- never written to.
REAL = {
    "baseline": ("1051", "latxa_llama31_8b_no_rag_extractive_mixed_eu_dev"),
    "retrieval": {
        "retrieve top1": ("1052", "latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev"),
        "retrieve top3": ("1053", "latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev"),
        "retrieve top5": ("1054", "latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev"),
        "rerank1": ("1055", "latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev"),
        "rerank3": ("1056", "latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev"),
        "rerank5": ("1057", "latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev"),
    },
    "fewshot_no_rag": ("1058", "latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev"),
    "row8": ("1059", "latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev"),
    "domain": {
        "SNS": ("1060", "latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev"),
        "Casi": ("1061", "latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev"),
    },
}


def smoke_id(real_prefix: str, real_base: str) -> tuple[str, str]:
    return f"{SMOKE_TAG}_{real_prefix}", real_base


def clean_up() -> None:
    for p in list(RUNS.glob(f"{SMOKE_TAG}_*")) :
        shutil.rmtree(p, ignore_errors=True)
    for p in list(METRICS.glob(f"{SMOKE_TAG}_*.json")):
        p.unlink()
    for p in list(CONFIG_DIR.glob(f"{SMOKE_TAG}_*.json")):
        p.unlink()


def make_smoke_config(real_prefix: str, real_base: str) -> tuple[str, str]:
    """Copy a real config to a smoke_stagedeval_ variant: --limit 4, self-feedback
    off (for speed), output redirected to a smoke run dir. Never touches the real
    config file or its output path."""
    real_path = CONFIG_DIR / f"{real_prefix}_{real_base}.json"
    cfg = json.loads(real_path.read_text())
    cfg["limit"] = LIMIT
    cfg["self_feedback"] = False
    s_prefix, s_base = smoke_id(real_prefix, real_base)
    cfg["output"] = f"experiments/runs/{s_prefix}_{s_base}/predictions.jsonl"
    smoke_path = CONFIG_DIR / f"{s_prefix}_{s_base}.json"
    smoke_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return s_prefix, s_base


def build_smoke_spec() -> dict:
    spec: dict = {"lang": "eu"}
    spec["baseline"] = make_smoke_config(*REAL["baseline"])
    spec["retrieval"] = {label: make_smoke_config(*rb) for label, rb in REAL["retrieval"].items()}
    spec["fewshot_no_rag"] = make_smoke_config(*REAL["fewshot_no_rag"])
    spec["row8"] = make_smoke_config(*REAL["row8"])
    spec["domain"] = {corpus: make_smoke_config(*rb) for corpus, rb in REAL["domain"].items()}
    return spec


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise SystemExit(f"SMOKE TEST FAILED: {message}")


def main() -> None:
    print(f"=== smoke test: staged eval approach (Latxa, limit={LIMIT} records/config) ===")
    clean_up()
    spec = build_smoke_spec()

    # ── Stage A: infer + evaluate the baseline + 6-config retrieval sweep ──────
    print("\n--- stage A: baseline + retrieval sweep ---")
    stage_a_labeled = [("baseline", spec["baseline"])] + list(spec["retrieval"].items())
    stage_a = [rb for _, rb in stage_a_labeled]
    job = submit_infer_array(stage_a, "smoketest_A")
    wait_for_job(job)
    evaluate(stage_a, spec["lang"])

    for label, (prefix, base) in stage_a_labeled:
        pred = RUNS / f"{prefix}_{base}_seed42" / "predictions.jsonl"
        check(pred.exists() and len(pred.read_text().splitlines()) == LIMIT,
              f"stage A '{label}' produced {LIMIT} real predictions ({prefix}_{base}_seed42)")

    win_r, scores_r = best_by_meanq(spec["retrieval"])
    check(win_r is not None, "stage A MeanQ produced a winner from the retrieval sweep")
    print(f"  stage A winner: {win_r}  (MeanQ scores: {scores_r})")

    # ── The central claim: row 8 is wired to the winner BEFORE it runs ─────────
    print("\n--- wiring row 8 to the stage-A winner (before inference) ---")
    win_prefix, win_base = spec["retrieval"][win_r]
    winner_fields = base_retrieval_fields(win_prefix, win_base)
    r8_prefix, r8_base = spec["row8"]
    row8_path = CONFIG_DIR / f"{r8_prefix}_{r8_base}.json"
    before = json.loads(row8_path.read_text())
    apply_base(r8_prefix, r8_base, winner_fields, apply=True)
    after = json.loads(row8_path.read_text())
    check(
        all(after.get(k) == v for k, v in winner_fields.items()),
        f"row 8's config file was rewritten with the winner's retrieval fields {winner_fields} "
        f"BEFORE stage B submission (was {({k: before.get(k) for k in winner_fields})})",
    )

    # ── Stage B: infer + evaluate row 7 (fewshot, no wiring needed) + row 8 ────
    print("\n--- stage B: few-shot no-RAG + few-shot+best-RAG (row 8) ---")
    stage_b_labeled = [("3shot-noRAG", spec["fewshot_no_rag"]), ("row8", spec["row8"])]
    stage_b = [rb for _, rb in stage_b_labeled]
    job = submit_infer_array(stage_b, "smoketest_B")
    wait_for_job(job)
    evaluate(stage_b, spec["lang"])

    for label, (prefix, base) in stage_b_labeled:
        pred = RUNS / f"{prefix}_{base}_seed42" / "predictions.jsonl"
        check(pred.exists() and len(pred.read_text().splitlines()) == LIMIT,
              f"stage B '{label}' produced {LIMIT} real predictions ({prefix}_{base}_seed42)")

    pre_domain = dict(spec["retrieval"])
    pre_domain["3-shot + best RAG"] = spec["row8"]
    win_pd, scores_pd = best_by_meanq(pre_domain)
    check(win_pd is not None, "stage B decision (rows 1-6 + row 8) produced a domain-row winner")
    print(f"  domain base winner: {win_pd}  (MeanQ scores: {scores_pd})")

    # ── Wire + run stage C: domain restriction rows ────────────────────────────
    print("\n--- wiring domain rows to the stage-B winner (before inference) ---")
    pd_prefix, pd_base = pre_domain[win_pd]
    domain_fields = base_retrieval_fields(pd_prefix, pd_base)
    domain_fields = {k: v for k, v in domain_fields.items() if k != "retrieval_index"}
    for corpus, (dprefix, dbase) in spec["domain"].items():
        apply_base(dprefix, dbase, domain_fields, apply=True)
        cfg = json.loads((CONFIG_DIR / f"{dprefix}_{dbase}.json").read_text())
        check(
            all(cfg.get(k) == v for k, v in domain_fields.items()),
            f"domain row '{corpus}' rewritten with {domain_fields} before stage C submission",
        )

    print("\n--- stage C: domain restriction (SNS-only, CasiMedicos-only) ---")
    stage_c = list(spec["domain"].values())
    job = submit_infer_array(stage_c, "smoketest_C")
    wait_for_job(job)
    evaluate(stage_c, spec["lang"])

    for corpus, (prefix, base) in spec["domain"].items():
        pred = RUNS / f"{prefix}_{base}_seed42" / "predictions.jsonl"
        check(pred.exists() and len(pred.read_text().splitlines()) == LIMIT,
              f"stage C domain row '{corpus}' produced {LIMIT} real predictions ({prefix}_{base}_seed42)")

    print("\n=== SMOKE TEST PASSED: staged infer -> eval -> decide -> infer works end to end ===")
    print(f"Evidence retained under experiments/runs/{SMOKE_TAG}_*, "
          f"reports/metrics/{SMOKE_TAG}_*.json, configs/experiments/{SMOKE_TAG}_*.json")


if __name__ == "__main__":
    main()
