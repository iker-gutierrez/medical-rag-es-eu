#!/usr/bin/env python
"""Generate 132 Qwen/Qwen3.5-9B Spanish dev configs (11 conditions × 3 dev sets × 2 think modes × 2 SF modes).

noSF IDs 316-381, SF IDs 382-447.
  316-337: SNS1064   (11 no_think + 11 think)
  338-359: CasiMedicos (11 no_think + 11 think)
  360-381: mixed       (11 no_think + 11 think)
  382-403: SNS1064 SF
  404-425: CasiMedicos SF
  426-447: mixed SF
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs/experiments"
MODEL = "Qwen/Qwen3.5-9B"
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

SNS_EXPERIMENTS = [
    ("no_rag", {"retrieval_top_k": 0, "few_shot_k": 0}),
    ("rag_e5_topk1", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 1}),
    ("rag_e5_topk3", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 3}),
    ("rag_e5_topk5", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 5}),
    ("rag_e5_rerank1", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 1, "reranker_device": "cpu"}),
    ("rag_e5_rerank3", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 3, "reranker_device": "cpu"}),
    ("rag_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("3shot_no_rag", {"retrieval_top_k": 0, "few_shot_file": "data/processed/sns1064/train.jsonl", "few_shot_k": 3, "few_shot_mode": "random"}),
    ("rag_cross_domain_e5_rerank5", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("rag_mixed_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("rag_3shot_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu", "few_shot_file": "data/processed/sns1064/train.jsonl", "few_shot_k": 3, "few_shot_mode": "random"}),
]

CASI_EXPERIMENTS = [
    ("no_rag", {"retrieval_top_k": 0, "few_shot_k": 0}),
    ("rag_e5_topk1", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 1}),
    ("rag_e5_topk3", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 3}),
    ("rag_e5_topk5", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 5}),
    ("rag_e5_rerank1", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 1, "reranker_device": "cpu"}),
    ("rag_e5_rerank3", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 3, "reranker_device": "cpu"}),
    ("rag_e5_rerank5", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("3shot_no_rag", {"retrieval_top_k": 0, "few_shot_file": "data/processed/casimedicos/train.jsonl", "few_shot_k": 3, "few_shot_mode": "random"}),
    ("rag_cross_domain_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("rag_mixed_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("rag_3shot_e5_rerank5", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu", "few_shot_file": "data/processed/casimedicos/train.jsonl", "few_shot_k": 3, "few_shot_mode": "random"}),
]

MIXED_EXPERIMENTS = [
    ("no_rag", {"retrieval_top_k": 0, "few_shot_k": 0}),
    ("rag_e5_topk1", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 1}),
    ("rag_e5_topk3", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 3}),
    ("rag_e5_topk5", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 5}),
    ("rag_e5_rerank1", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 1, "reranker_device": "cpu"}),
    ("rag_e5_rerank3", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 3, "reranker_device": "cpu"}),
    ("rag_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("3shot_no_rag", {"retrieval_top_k": 0, "few_shot_file": "data/processed/sns1064_casimedicos/train.jsonl", "few_shot_k": 3, "few_shot_mode": "random"}),
    ("rag_sns1064_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("rag_casimedicos_e5_rerank5", {"retrieval_index": "models/retrieval/casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu"}),
    ("rag_3shot_e5_rerank5", {"retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large", "retrieval_top_k": 15, "reranker_model": RERANKER, "reranker_top_k": 5, "reranker_device": "cpu", "few_shot_file": "data/processed/sns1064_casimedicos/train.jsonl", "few_shot_k": 3, "few_shot_mode": "random"}),
]

DEV_SETS = [
    ("sns1064",     "data/processed/sns1064/dev.jsonl",                  SNS_EXPERIMENTS),
    ("casimedicos", "data/processed/casimedicos/dev.jsonl",               CASI_EXPERIMENTS),
    ("mixed",       "data/processed/sns1064_casimedicos/dev.jsonl",       MIXED_EXPERIMENTS),
]

NOSF_START = 316
SF_OFFSET  = 66   # 316+66=382


def make_nosf(run_id, dev_slug, dev_input, exp_slug, partial, think):
    mode = "think" if think else "no_think"
    name = f"qwen35_9b_{exp_slug}_{mode}_extractive_{dev_slug}"
    cfg = {
        "experiment_name": name,
        "input": dev_input,
        "output": f"experiments/runs/{run_id}_{name}_dev/predictions.jsonl",
        "model": MODEL,
        "prompt_style": "extractive",
        "think": think,
        "self_feedback": False,
        "temperature": 0.0,
        "trust_remote_code": True,
        "max_new_tokens": 1024,
    }
    cfg.update(partial)
    return name, cfg


def make_sf(nosf_id, dev_slug, dev_input, exp_slug, partial, think):
    sf_id = nosf_id + SF_OFFSET
    mode = "think" if think else "no_think"
    name = f"qwen35_9b_{exp_slug}_{mode}_extractive_{dev_slug}_sf"
    nosf_name = f"qwen35_9b_{exp_slug}_{mode}_extractive_{dev_slug}"
    cfg = {
        "experiment_name": name,
        "input": dev_input,
        "output": f"experiments/runs/{sf_id}_{name}_dev/predictions.jsonl",
        "initial_predictions": f"experiments/runs/{nosf_id}_{nosf_name}_dev/predictions.jsonl",
        "model": MODEL,
        "prompt_style": "extractive",
        "think": think,
        "self_feedback": True,
        "temperature": 0.0,
        "trust_remote_code": True,
        "max_new_tokens": 1024,
        "feedback_max_new_tokens": 512,
    }
    cfg.update(partial)
    return sf_id, name, cfg


def main():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    nosf_id = NOSF_START
    nosf_configs = []

    for dev_slug, dev_input, experiments in DEV_SETS:
        for think in (False, True):
            for exp_slug, partial in experiments:
                name, cfg = make_nosf(nosf_id, dev_slug, dev_input, exp_slug, partial, think)
                path = CFG_DIR / f"{nosf_id}_{name}_dev.json"
                path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
                nosf_configs.append((nosf_id, dev_slug, dev_input, exp_slug, partial, think))
                print(f"  {nosf_id}: {name}")
                nosf_id += 1

    for nosf_id, dev_slug, dev_input, exp_slug, partial, think in nosf_configs:
        sf_id, name, cfg = make_sf(nosf_id, dev_slug, dev_input, exp_slug, partial, think)
        path = CFG_DIR / f"{sf_id}_{name}_dev.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"  {sf_id}: {name}")

    print(f"\nWrote 132 configs: noSF {NOSF_START}-{NOSF_START+65}, SF {NOSF_START+SF_OFFSET}-{NOSF_START+SF_OFFSET+65}")


if __name__ == "__main__":
    main()
