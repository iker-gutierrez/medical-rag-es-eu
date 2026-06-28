#!/usr/bin/env python
"""Generate 66 Qwen/Qwen3-8B Spanish dev configs (11 conditions × 3 dev sets × 2 modes).

Run IDs 184-249.
  184-205: SNS1064     (11 no_think + 11 think)
  206-227: CasiMedicos (11 no_think + 11 think)
  228-249: mixed       (11 no_think + 11 think)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs/experiments"
MODEL = "Qwen/Qwen3-8B"

RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Each entry: (slug, partial_config)
# partial_config is merged into the base; think/self_feedback/output filled in per mode.
SNS_EXPERIMENTS = [
    ("no_rag", {
        "retrieval_top_k": 0, "few_shot_k": 0,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk1", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 1,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk3", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 3,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk5", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 5,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank1", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 1, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank3", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 3, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("3shot_no_rag", {
        "retrieval_top_k": 0,
        "few_shot_file": "data/processed/sns1064/train.jsonl",
        "few_shot_k": 3, "few_shot_mode": "random",
        "max_new_tokens": 1024,
    }),
    ("rag_cross_domain_e5_rerank5", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_mixed_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_3shot_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "few_shot_file": "data/processed/sns1064/train.jsonl",
        "few_shot_k": 3, "few_shot_mode": "random",
        "max_new_tokens": 1024,
    }),
]

CASI_EXPERIMENTS = [
    ("no_rag", {
        "retrieval_top_k": 0, "few_shot_k": 0,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk1", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 1,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk3", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 3,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk5", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 5,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank1", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 1, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank3", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 3, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank5", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("3shot_no_rag", {
        "retrieval_top_k": 0,
        "few_shot_file": "data/processed/casimedicos/train.jsonl",
        "few_shot_k": 3, "few_shot_mode": "random",
        "max_new_tokens": 1024,
    }),
    ("rag_cross_domain_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_mixed_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_3shot_e5_rerank5", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "few_shot_file": "data/processed/casimedicos/train.jsonl",
        "few_shot_k": 3, "few_shot_mode": "random",
        "max_new_tokens": 1024,
    }),
]

MIXED_EXPERIMENTS = [
    ("no_rag", {
        "retrieval_top_k": 0, "few_shot_k": 0,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk1", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 1,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk3", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 3,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_topk5", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 5,
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank1", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 1, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank3", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 3, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("3shot_no_rag", {
        "retrieval_top_k": 0,
        "few_shot_file": "data/processed/sns1064_casimedicos/train.jsonl",
        "few_shot_k": 3, "few_shot_mode": "random",
        "max_new_tokens": 1024,
    }),
    ("rag_sns1064_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_casimedicos_e5_rerank5", {
        "retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "max_new_tokens": 1024,
    }),
    ("rag_3shot_e5_rerank5", {
        "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
        "retrieval_top_k": 15,
        "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu",
        "few_shot_file": "data/processed/sns1064_casimedicos/train.jsonl",
        "few_shot_k": 3, "few_shot_mode": "random",
        "max_new_tokens": 1024,
    }),
]

DEV_SETS = [
    ("sns1064", "data/processed/sns1064/dev.jsonl", SNS_EXPERIMENTS),
    ("casimedicos", "data/processed/casimedicos/dev.jsonl", CASI_EXPERIMENTS),
    ("mixed", "data/processed/sns1064_casimedicos/dev.jsonl", MIXED_EXPERIMENTS),
]

THINK_MODES = [False, True]  # no_think first, then think

START_ID = 184


def make_config(
    run_id: int,
    dev_slug: str,
    dev_input: str,
    exp_slug: str,
    partial: dict,
    think: bool,
) -> dict:
    mode = "think" if think else "no_think"
    exp_name = f"qwen3_8b_{exp_slug}_{mode}_extractive_{dev_slug}"
    cfg = {
        "experiment_name": exp_name,
        "input": dev_input,
        "output": f"experiments/runs/{run_id}_{exp_name}_dev/predictions.jsonl",
        "model": MODEL,
        "prompt_style": "extractive",
        "think": think,
        "self_feedback": False,
        "temperature": 0.0,
        "trust_remote_code": True,
    }
    cfg.update(partial)
    return cfg


def main() -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = START_ID
    written = []
    for dev_slug, dev_input, experiments in DEV_SETS:
        for think in THINK_MODES:
            for exp_slug, partial in experiments:
                cfg = make_config(run_id, dev_slug, dev_input, exp_slug, partial, think)
                out = CFG_DIR / f"{run_id}_{cfg['experiment_name']}_dev.json"
                out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
                written.append(out.name)
                print(f"  {run_id}: {cfg['experiment_name']}")
                run_id += 1
    print(f"\nWrote {len(written)} configs (IDs {START_ID}–{run_id - 1})")


if __name__ == "__main__":
    main()
