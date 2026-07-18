#!/usr/bin/env python
"""Run Latxa's rows 8 (fewshot), 9 (SNS domain), 10 (CasiMedicos domain) at
their correct base config -- retrieve top 1 -- staged in two waves rather than one
flat batch, so row 8 finishes and evaluates before rows 9/10 start.

This is NOT a "does row 8 win, then wire 9/10 to it" decision the way
staged_ablation_runner.py's stage B->C wiring is: row 8, 9, and 10 are three
independent forks off the SAME already-decided base (Latxa's own MeanQ winner,
retrieve top 1 -- see write_result_tables.py's FORCED_REFERENCES comment), not a
chain where a later row's config depends on an earlier row's result. Row 8
adds few-shot demonstrations; rows 9/10 restrict the retrieval corpus -- mixing
them (building 9/10 on top of row 8) would confound few-shot with corpus
restriction, which is not what "does the corpus matter" is supposed to isolate.

The staging here is purely about NOT wasting GPU time: there is no reason to
wait for row 8 before starting 9/10 except that this script runs them as two
waves for a clear go/no-go checkpoint (confirm row 8's predictions look sane
before committing the rest of the budget). If that checkpoint isn't wanted,
just run all three configs as one wave instead.

All three configs (ids 1059/1060/1061) are ALREADY rewired to retrieval_top_k=1
by this session's earlier work -- this script only submits inference and
evaluation, it does not touch any config file.

Usage:
  python scripts/run_latxa_row8_9_10.py            # submit + wait + evaluate, both waves
  python scripts/run_latxa_row8_9_10.py --dry-run   # print the plan, run nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from staged_ablation_runner import submit_infer_array, wait_for_job, evaluate  # noqa: E402

ROW8 = [("1059", "latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev")]
ROWS_9_10 = [
    ("1060", "latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev"),
    ("1061", "latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Wave 1 (row 8, fewshot):", ROW8)
        print("Wave 2 (rows 9-10, domain):", ROWS_9_10)
        print("3 seeds each (42, 43, 44) -> 3 + 6 = 9 inference tasks total, 1 GPU/task, 2 concurrent.")
        return

    print("=== Wave 1: row 8 (fewshot, Latxa retrieve-top1) ===")
    job = submit_infer_array(ROW8, "latxa_row8")
    wait_for_job(job)
    evaluate(ROW8, "eu")

    print("\n=== Wave 2: rows 9-10 (domain restriction, Latxa retrieve-top1) ===")
    job = submit_infer_array(ROWS_9_10, "latxa_row9_10")
    wait_for_job(job)
    evaluate(ROWS_9_10, "eu")

    print("\nDone. Next: python scripts/write_result_tables.py to regenerate the "
          "manuscript tables, then recompile with latexmk, then update the "
          "eu_dev_ablation_results.md report.")


if __name__ == "__main__":
    main()
