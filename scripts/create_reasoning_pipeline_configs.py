#!/usr/bin/env python
"""Emit the reasoning-pipeline configs for Spanish and Basque.

Retrieval is frozen to the winning row of each dev decision table, so any
difference from those rows is attributable to the reasoning pipeline alone:

  ES  Qwen3.5-9B (think), retrieve top-15 -> rerank top-5   (es_dev_ablation_results.md)
  EU  Llama-3.1-8B-Instruct,  retrieve top-15 -> rerank top-5 (eu_dev_ablation_results.md)

Sampling params are inherited from the frozen runs' own v2 configs (1276 / 1046).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"

# The single-pass rows these pipelines are measured against: the winning row of
# each dev decision table. ES = rerank top 5 (think); EU = retrieve top 3.
ES_BASELINE_RUN = "1276_qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev_v2"
EU_BASELINE_RUN = "1042_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev_v2"

ES_BASE = {
    "model": "Qwen/Qwen3.5-9B",
    "language": "es",
    "input": "data/processed/sns1064_casimedicos/dev.jsonl",
    "think": True,
    "trust_remote_code": True,
    "prompt_style": "extractive",
    # sampling, from config 1276 (v2)
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.05,
    "presence_penalty": 0.1,
    # Token budgets are the ablation's own (config 1276 v2), not reduced. An earlier
    # draft cut them to 6144/4096 to make the multi-round pipelines cheaper, but that
    # silently handicaps them: a pipeline truncated at a budget the baseline never hit
    # would lose on generation length rather than on reasoning quality.
    "max_model_len": 31744,
    "reasoning_parser": "qwen3",
    "thinking_token_budget": 20700,
    "max_new_tokens": 22450,
    "thought_max_new_tokens": 22450,
    "repetition_detection_max_pattern": 20,
    "repetition_detection_min_pattern": 1,
    "repetition_detection_min_count": 8,
    # retrieval, frozen
    "retrieval_index": "models/retrieval/sns1064_casimedicos_train_multilingual_e5_large",
    "retrieval_top_k": 15,
    "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "reranker_top_k": 5,
    "reranker_device": "cpu",
}

EU_BASE = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "language": "eu",
    "input": "data/processed/sns1064_casimedicos_eu/dev.jsonl",
    "think": False,
    "prompt_style": "extractive",
    # sampling, from config 1042 (v2)
    "temperature": 0.6,
    "top_p": 0.9,
    "min_p": 0.05,
    "presence_penalty": 0.1,
    "max_model_len": 16384,
    "max_new_tokens": 1750,
    "thought_max_new_tokens": 1024,
    "repetition_detection_max_pattern": 20,
    "repetition_detection_min_pattern": 1,
    "repetition_detection_min_count": 8,
    # Retrieval, frozen: **retrieve top 3, no reranker**. The prose under the EU decision
    # table names "rerank top 5" because it ranks on BERTScore alone, but the table
    # itself makes retrieve top 3 the better row on every other axis: BERT-MC mean 61.53 vs
    # 55.87 and MC accuracy 51.85 vs 39.68, for a 0.86 BERTScore loss -- the same
    # trade-off the ES table used to pick think mode. It is also cheaper (no
    # cross-encoder pass, ~1562 vs 2235 tokens/sample).
    "retrieval_index": "models/retrieval/sns1064_casimedicos_eu_train_multilingual_e5_large",
    "retrieval_top_k": 3,
    "reranker_model": "",
    "reranker_top_k": 0,
}

# (config id, pipeline, name suffix, pipeline-specific overrides)
PIPELINES = [
    (0, "structured_cot", "structured_cot", {}),
    (1, "thought_rag", "thought_rag", {"thought_query_mode": "question_plus_thought"}),
    (2, "thought_rag_iter", "thought_rag_iter", {"rounds": 2, "max_context_docs": 8}),
    (
        3,
        "marag",
        "marag",
        {
            "rounds": 3,
            "num_candidates": 3,
            # Any multiple-choice disagreement at all is a conflict (MA-RAG's own
            # criterion); for open answers, >0.15 mean pairwise cosine distance.
            "conflict_threshold": 0.15,
            "max_context_docs": 10,
            "query_max_new_tokens": 128,
            "ranking_max_new_tokens": 32,
        },
    ),
]

# retrieval_tag goes into the run name, so a run dir never claims a retrieval
# setting it does not use (the EU runs are retrieve top 3, not rerank top 5).
FAMILIES = [
    (1300, "qwen35_9b", "mixed", "e5_rerank5", ES_BASE, ES_BASELINE_RUN),
    (1310, "llama31_8b", "mixed_eu", "e5_topk3", EU_BASE, EU_BASELINE_RUN),
]


def main() -> None:
    written = []
    for base_id, model_tag, task_tag, retrieval_tag, base, baseline_run in FAMILIES:
        for offset, pipeline, suffix, overrides in PIPELINES:
            config_id = base_id + offset
            name = f"{model_tag}_{suffix}_{retrieval_tag}_extractive_{task_tag}_dev"
            run_dir = f"{config_id}_{name}"
            payload = {
                "experiment_name": name,
                "pipeline": pipeline,
                "baseline_run": baseline_run,
                "output": f"experiments/runs/{run_dir}/predictions.jsonl",
                **base,
                **overrides,
            }
            path = CONFIG_DIR / f"{run_dir}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(path.name)

    for name in written:
        print(name)


if __name__ == "__main__":
    main()
