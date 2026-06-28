#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path("configs/experiments")

MODELS = [
    ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct", False),
    ("latxa_llama31_8b", "HiTZ/Latxa-Llama-3.1-8B-Instruct", True),
]

DATASETS = [
    {
        "slug": "sns1064_eu",
        "input": "data/processed/sns1064_eu/dev.jsonl",
        "train": "data/processed/sns1064_eu/train.jsonl",
        "indices": {
            "in_domain": "models/retrieval/sns1064_eu_train_multilingual_e5_large",
            "cross_domain": "models/retrieval/casimedicos_eu_train_multilingual_e5_large",
            "mixed": "models/retrieval/sns1064_casimedicos_eu_train_multilingual_e5_large",
        },
    },
    {
        "slug": "casimedicos_eu",
        "input": "data/processed/casimedicos_eu/dev.jsonl",
        "train": "data/processed/casimedicos_eu/train.jsonl",
        "indices": {
            "in_domain": "models/retrieval/casimedicos_eu_train_multilingual_e5_large",
            "cross_domain": "models/retrieval/sns1064_eu_train_multilingual_e5_large",
            "mixed": "models/retrieval/sns1064_casimedicos_eu_train_multilingual_e5_large",
        },
    },
    {
        "slug": "mixed_eu",
        "input": "data/processed/sns1064_casimedicos_eu/dev.jsonl",
        "train": "data/processed/sns1064_casimedicos_eu/train.jsonl",
        "indices": {
            "in_domain": "models/retrieval/sns1064_casimedicos_eu_train_multilingual_e5_large",
            "cross_domain_sns": "models/retrieval/sns1064_eu_train_multilingual_e5_large",
            "cross_domain_casimedicos": "models/retrieval/casimedicos_eu_train_multilingual_e5_large",
        },
    },
]


def base_config(model_slug: str, model_id: str, trust_remote_code: bool, dataset: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": model_id,
        "language": "eu",
        "prompt_style": "extractive",
        "think": False,
        "self_feedback": True,
        "max_new_tokens": 256,
        "feedback_max_new_tokens": 256,
        "temperature": 0.0,
        "input": dataset["input"],
    }
    if trust_remote_code:
        config["trust_remote_code"] = True
    return config


def retrieval_config(index: str, retrieval_top_k: int, reranker_top_k: int = 0) -> dict[str, Any]:
    config: dict[str, Any] = {
        "retrieval_index": index,
        "retrieval_top_k": retrieval_top_k,
    }
    if reranker_top_k > 0:
        config.update(
            {
                "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                "reranker_top_k": reranker_top_k,
                "reranker_device": "cpu",
            }
        )
    return config


def experiment_specs(dataset: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    indices = dataset["indices"]
    if dataset["slug"] == "mixed_eu":
        cross_a = ("rag_sns1064_e5_rerank5", indices["cross_domain_sns"])
        cross_b = ("rag_casimedicos_e5_rerank5", indices["cross_domain_casimedicos"])
    else:
        cross_a = ("rag_cross_domain_e5_rerank5", indices["cross_domain"])
        cross_b = ("rag_mixed_e5_rerank5", indices["mixed"])

    return [
        ("no_rag", {"few_shot_k": 0, "retrieval_top_k": 0}),
        ("rag_e5_topk1", retrieval_config(indices["in_domain"], 1)),
        ("rag_e5_topk3", retrieval_config(indices["in_domain"], 3)),
        ("rag_e5_topk5", retrieval_config(indices["in_domain"], 5)),
        ("rag_e5_rerank1", retrieval_config(indices["in_domain"], 15, reranker_top_k=1)),
        ("rag_e5_rerank3", retrieval_config(indices["in_domain"], 15, reranker_top_k=3)),
        ("rag_e5_rerank5", retrieval_config(indices["in_domain"], 15, reranker_top_k=5)),
        (
            "3shot_no_rag",
            {
                "few_shot_file": dataset["train"],
                "few_shot_k": 3,
                "few_shot_mode": "random",
                "retrieval_top_k": 0,
            },
        ),
        (
            "rag_3shot_e5_rerank5",
            {
                "few_shot_file": dataset["train"],
                "few_shot_k": 3,
                "few_shot_mode": "random",
                **retrieval_config(indices["in_domain"], 15, reranker_top_k=5),
            },
        ),
        (cross_a[0], retrieval_config(cross_a[1], 15, reranker_top_k=5)),
        (cross_b[0], retrieval_config(cross_b[1], 15, reranker_top_k=5)),
    ]


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = 106
    created = []
    for dataset in DATASETS:
        for model_slug, model_id, trust_remote_code in MODELS:
            for experiment_slug, experiment_config in experiment_specs(dataset):
                name = f"{model_slug}_{experiment_slug}_extractive_{dataset['slug']}"
                config = base_config(model_slug, model_id, trust_remote_code, dataset)
                config.update(experiment_config)
                config["experiment_name"] = name
                config["output"] = f"experiments/runs/{run_id}_{name}_dev/predictions.jsonl"
                path = CONFIG_DIR / f"{run_id}_{name}_dev.json"
                path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                created.append(path)
                run_id += 1
    print(f"Wrote {len(created)} configs: {created[0]} .. {created[-1]}")


if __name__ == "__main__":
    main()
