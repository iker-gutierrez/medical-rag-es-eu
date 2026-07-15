#!/usr/bin/env python
"""Wire the dependent ablation configs to the MeanQ-best RAG base, per model.

The ablation grid has an intended dependency structure that the config files did not
enforce: the few-shot-plus-RAG row is meant to combine few-shot with *the best
retrieval configuration*, and the two domain-restriction rows are meant to restrict
the corpus of *the best configuration so far*. Both were hardcoded to rerank-top-5.
If MeanQ selects a different retrieval setting -- and for Llama it selects e5 top-5,
not rerank-5 -- those rows are built on the wrong base and the staged comparison is
invalid.

This script recomputes the dependency from MeanQ and rewrites the affected configs:

  row 8  (few-shot + RAG)          <- best of the six retrieval rows (1-6)
  rows 9,10 (domain restriction)   <- best of everything before them (rows 1-8),
                                      which includes few-shot with and without RAG

"Best" is highest MeanQ (see scripts/meanq.py). The retrieval fields
(retrieval_top_k, reranker_model, reranker_top_k) are copied from the winning
config; the row's own distinguishing field (few_shot_k for row 8, retrieval_index
for the domain rows) is preserved.

Run AFTER the fresh evaluation, so MeanQ is computed on the corrected numbers. It
reports every decision, and only rewrites a config when the base actually changes;
configs whose hardcoded base already matches the MeanQ winner are left untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from meanq import best_by_meanq, meanq  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "experiments"

# The retrieval fields that define a RAG base. Copied from the winner into dependents.
BASE_FIELDS = ("retrieval_top_k", "reranker_model", "reranker_top_k", "retrieval_index")

# Per model: the id/base of each grid row. `retrieval` = rows 1-6 (few-shot's pool);
# `pre_domain` = rows 1-8 (the domain rows' pool); `dependents` = the configs to fix.
# retrieval_index is intentionally NOT copied into the domain rows (they define their
# own restricted index); it IS copied into row 8 (which shares the mixed corpus).
MODELS = {
    "mistral7b": {
        "baseline": ("1260", "mistral7b_no_rag_no_think_extractive_mixed_dev"),
        "lang": "es", "task": "mixed_dev", "tag": "no_think_extractive",
        "retrieval": {
            "e5 top1": ("1261", "mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev"),
            "e5 top3": ("1262", "mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev"),
            "e5 top5": ("1263", "mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev"),
            "rerank1": ("1264", "mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev"),
            "rerank3": ("1265", "mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev"),
            "rerank5": ("1266", "mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev"),
        },
        "fewshot_no_rag": ("1267", "mistral7b_3shot_no_rag_no_think_extractive_mixed_dev"),
        "row8": ("1270", "mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev"),
        "domain": {
            "SNS": ("1268", "mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev"),
            "Casi": ("1269", "mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev"),
        },
    },
    "qwen35_9b_no_think": {
        "baseline": ("1128", "qwen35_9b_no_rag_no_think_extractive_mixed_dev"),
        "lang": "es", "task": "mixed_dev", "tag": "no_think_extractive",
        "retrieval": {
            "e5 top1": ("1129", "qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev"),
            "e5 top3": ("1130", "qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev"),
            "e5 top5": ("1131", "qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev"),
            "rerank1": ("1132", "qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev"),
            "rerank3": ("1133", "qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev"),
            "rerank5": ("1134", "qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev"),
        },
        "fewshot_no_rag": ("1135", "qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev"),
        "row8": ("1138", "qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev"),
        "domain": {
            "SNS": ("1136", "qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev"),
            "Casi": ("1137", "qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev"),
        },
    },
    "qwen35_9b_think": {
        "baseline": ("1270", "qwen35_9b_no_rag_think_extractive_mixed_dev"),
        "lang": "es", "task": "mixed_dev", "tag": "think_extractive",
        "retrieval": {
            "e5 top1": ("1271", "qwen35_9b_rag_e5_topk1_think_extractive_mixed_dev"),
            "e5 top3": ("1272", "qwen35_9b_rag_e5_topk3_think_extractive_mixed_dev"),
            "e5 top5": ("1273", "qwen35_9b_rag_e5_topk5_think_extractive_mixed_dev"),
            "rerank1": ("1274", "qwen35_9b_rag_e5_rerank1_think_extractive_mixed_dev"),
            "rerank3": ("1275", "qwen35_9b_rag_e5_rerank3_think_extractive_mixed_dev"),
            "rerank5": ("1276", "qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev"),
        },
        "fewshot_no_rag": ("1277", "qwen35_9b_3shot_no_rag_think_extractive_mixed_dev"),
        "row8": ("1280", "qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev"),
        "domain": {
            "SNS": ("1278", "qwen35_9b_rag_sns1064_e5_rerank5_think_extractive_mixed_dev"),
            "Casi": ("1279", "qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev"),
        },
    },
    "llama31_8b": {
        "baseline": ("1040", "llama31_8b_no_rag_extractive_mixed_eu_dev"),
        "lang": "eu", "task": "mixed_eu_dev", "tag": "extractive",
        "retrieval": {
            "e5 top1": ("1041", "llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev"),
            "e5 top3": ("1042", "llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev"),
            "e5 top5": ("1043", "llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev"),
            "rerank1": ("1044", "llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev"),
            "rerank3": ("1045", "llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev"),
            "rerank5": ("1046", "llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev"),
        },
        "fewshot_no_rag": ("1047", "llama31_8b_3shot_no_rag_extractive_mixed_eu_dev"),
        "row8": ("1048", "llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev"),
        "domain": {
            "SNS": ("1049", "llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev"),
            "Casi": ("1050", "llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev"),
        },
    },
    "latxa": {
        "baseline": ("1051", "latxa_llama31_8b_no_rag_extractive_mixed_eu_dev"),
        "lang": "eu", "task": "mixed_eu_dev", "tag": "extractive",
        "retrieval": {
            "e5 top1": ("1052", "latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev"),
            "e5 top3": ("1053", "latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev"),
            "e5 top5": ("1054", "latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev"),
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
    },
}


def base_retrieval_fields(prefix: str, base: str) -> dict:
    cfg = json.loads((CONFIG_DIR / f"{prefix}_{base}.json").read_text())
    return {k: cfg.get(k) for k in ("retrieval_top_k", "reranker_model", "reranker_top_k")}


def apply_base(target_prefix: str, target_base: str, fields: dict, apply: bool) -> bool:
    """Copy retrieval fields into the target config. Returns True if it changed."""
    path = CONFIG_DIR / f"{target_prefix}_{target_base}.json"
    cfg = json.loads(path.read_text())
    changed = {k: (cfg.get(k), v) for k, v in fields.items() if cfg.get(k) != v}
    if not changed:
        return False
    if apply:
        cfg.update(fields)
        cfg["rag_base_source"] = "MeanQ-selected (rewire_dependent_configs.py)"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    for k, (old, new) in changed.items():
        print(f"      {k}: {old} -> {new}")
    return True


def process_model(name: str, spec: dict, apply: bool) -> list[str]:
    print(f"\n=== {name} ===")
    changed_runs: list[str] = []

    # Stage A: best of the six retrieval rows -> feeds row 8.
    win_r, scores_r = best_by_meanq(spec["retrieval"])
    if win_r is None:
        print("  no retrieval metrics yet -- skipping (run after evaluation)")
        return changed_runs
    print("  retrieval MeanQ:", ", ".join(f"{k}={v:.2f}" for k, v in
          sorted(scores_r.items(), key=lambda x: -x[1])))
    print(f"  best retrieval (row 8 base): {win_r}")

    win_prefix, win_base = spec["retrieval"][win_r]
    base_fields = base_retrieval_fields(win_prefix, win_base)
    r8_prefix, r8_base = spec["row8"]
    print(f"  row 8 ({r8_base}):")
    if apply_base(r8_prefix, r8_base, base_fields, apply):
        changed_runs.append(f"{r8_prefix}_{r8_base}")
    else:
        print("      already matches MeanQ winner")

    # Stage B: best of rows 0-8 -> feeds domain. Includes the baseline (row 0),
    # per the staged procedure, though a baseline win would be degenerate for a
    # domain-restriction row (nothing to restrict) and is flagged rather than applied.
    pre_domain = dict(spec["retrieval"])
    if "baseline" in spec:
        pre_domain["baseline"] = spec["baseline"]
    pre_domain["3-shot no RAG"] = spec["fewshot_no_rag"]
    pre_domain["3-shot + best RAG"] = spec["row8"]
    win_pd, scores_pd = best_by_meanq(pre_domain)
    print(f"  best of rows 0-8 (domain base): {win_pd}"
          f"  (MeanQ {scores_pd.get(win_pd, float('nan')):.2f})")

    if win_pd == "baseline":
        print("      NOTE: baseline won the domain pool -- domain restriction is not "
              "meaningful without retrieval; leaving domain rows unchanged.")
        return changed_runs
    pd_prefix, pd_base = pre_domain[win_pd]
    domain_base_fields = base_retrieval_fields(pd_prefix, pd_base)
    # The domain rows keep their own restricted retrieval_index; only the
    # top-k / reranker settings are inherited.
    domain_base_fields = {k: v for k, v in domain_base_fields.items()
                          if k != "retrieval_index"}
    for corpus, (dprefix, dbase) in spec["domain"].items():
        print(f"  domain {corpus} ({dbase}):")
        if apply_base(dprefix, dbase, domain_base_fields, apply):
            changed_runs.append(f"{dprefix}_{dbase}")
        else:
            print("      already matches MeanQ winner")

    return changed_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite configs. Without it, report only (dry run).")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    help="Subset of models to process.")
    ap.add_argument("--out", default="", help="Write the list of changed configs here.")
    args = ap.parse_args()

    changed: list[str] = []
    for name in args.models:
        if name not in MODELS:
            print(f"unknown model {name}"); continue
        changed += process_model(name, MODELS[name], args.apply)

    print(f"\n{'CHANGED' if args.apply else 'WOULD CHANGE'} {len(changed)} dependent configs:")
    for c in changed:
        print(f"  {c}")
    if not changed:
        print("  (none -- all dependent configs already use the MeanQ-best base)")
    if args.out and changed:
        Path(args.out).write_text("\n".join(changed) + "\n")
        print(f"\nwrote changed list to {args.out}")
    if not args.apply:
        print("\nDRY RUN -- rerun with --apply to rewrite the configs.")


if __name__ == "__main__":
    main()
