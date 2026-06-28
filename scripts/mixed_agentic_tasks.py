from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "metrics"


@dataclass(frozen=True)
class MixedAgenticTask:
    run_id: str
    dev_set: str
    language: str
    baseline_model: str
    judge_model_label: str
    judge_model: str
    baseline_pool: tuple[str, ...]
    candidate_pool: tuple[str, ...]
    trust_remote_code: bool = False
    max_new_tokens: int = 512


TASKS = (
    # Spanish: Mistral baseline, Qwen judge.
    MixedAgenticTask(
        run_id="172_mixed_agentic_es_mistral_baseline_qwen_judge_sns1064_dev",
        dev_set="SNS1064 dev",
        language="es",
        baseline_model="Mistral baseline",
        judge_model_label="Qwen Agentic Reasoner",
        judge_model="Qwen/Qwen3.5-4B",
        baseline_pool=("17_mistral7b_no_rag_no_think_extractive_sf_dev",),
        candidate_pool=(
            "32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev",
            "30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev",
            "33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev",
            "28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev",
            "31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev",
            "29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev",
            "20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev",
            "36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev",
            "37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev",
            "38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev",
        ),
        trust_remote_code=True,
    ),
    MixedAgenticTask(
        run_id="173_mixed_agentic_es_mistral_baseline_qwen_judge_casimedicos_dev",
        dev_set="CasiMedicos dev",
        language="es",
        baseline_model="Mistral baseline",
        judge_model_label="Qwen Agentic Reasoner",
        judge_model="Qwen/Qwen3.5-4B",
        baseline_pool=("39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev",),
        candidate_pool=tuple(
            f"{idx}_mistral7b_{suffix}"
            for idx, suffix in (
                (40, "rag_no_think_e5_topk1_extractive_sf_casimedicos_dev"),
                (41, "rag_no_think_e5_topk3_extractive_sf_casimedicos_dev"),
                (42, "rag_no_think_e5_topk5_extractive_sf_casimedicos_dev"),
                (43, "rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev"),
                (44, "rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev"),
                (45, "rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev"),
                (46, "random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev"),
                (47, "rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev"),
                (48, "rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev"),
                (49, "rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev"),
            )
        ),
        trust_remote_code=True,
    ),
    MixedAgenticTask(
        run_id="174_mixed_agentic_es_mistral_baseline_qwen_judge_mixed_dev",
        dev_set="SNS1064+CasiMedicos dev",
        language="es",
        baseline_model="Mistral baseline",
        judge_model_label="Qwen Agentic Reasoner",
        judge_model="Qwen/Qwen3.5-4B",
        baseline_pool=("50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev",),
        candidate_pool=tuple(
            f"{idx}_mistral7b_{suffix}"
            for idx, suffix in (
                (51, "rag_no_think_e5_topk1_extractive_sf_mixed_dev"),
                (52, "rag_no_think_e5_topk3_extractive_sf_mixed_dev"),
                (53, "rag_no_think_e5_topk5_extractive_sf_mixed_dev"),
                (54, "rag_no_think_e5_rerank1_extractive_sf_mixed_dev"),
                (55, "rag_no_think_e5_rerank3_extractive_sf_mixed_dev"),
                (56, "rag_no_think_e5_rerank5_extractive_sf_mixed_dev"),
                (57, "random_3shot_no_rag_no_think_extractive_sf_mixed_dev"),
                (58, "rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev"),
                (59, "rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev"),
                (60, "rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev"),
            )
        ),
        trust_remote_code=True,
    ),
    # Spanish: Qwen baseline, Llama judge.
    MixedAgenticTask(
        run_id="175_mixed_agentic_es_qwen_baseline_llama_judge_sns1064_dev",
        dev_set="SNS1064 dev",
        language="es",
        baseline_model="Qwen baseline",
        judge_model_label="Llama Agentic Reasoner",
        judge_model="meta-llama/Llama-3.1-8B-Instruct",
        baseline_pool=(
            "94_qwen35_4b_no_rag_no_think_extractive_sns1064_dev",
            "95_qwen35_4b_no_rag_think_extractive_sns1064_dev",
        ),
        candidate_pool=(
            "96_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_sns1064_dev",
            "97_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_sns1064_dev",
        ),
    ),
    MixedAgenticTask(
        run_id="176_mixed_agentic_es_qwen_baseline_llama_judge_casimedicos_dev",
        dev_set="CasiMedicos dev",
        language="es",
        baseline_model="Qwen baseline",
        judge_model_label="Llama Agentic Reasoner",
        judge_model="meta-llama/Llama-3.1-8B-Instruct",
        baseline_pool=(
            "98_qwen35_4b_no_rag_no_think_extractive_casimedicos_dev",
            "99_qwen35_4b_no_rag_think_extractive_casimedicos_dev",
        ),
        candidate_pool=(
            "100_qwen35_4b_rag_no_think_e5_topk3_extractive_casimedicos_dev",
            "101_qwen35_4b_rag_think_e5_topk3_extractive_casimedicos_dev",
        ),
    ),
    MixedAgenticTask(
        run_id="177_mixed_agentic_es_qwen_baseline_llama_judge_mixed_dev",
        dev_set="SNS1064+CasiMedicos dev",
        language="es",
        baseline_model="Qwen baseline",
        judge_model_label="Llama Agentic Reasoner",
        judge_model="meta-llama/Llama-3.1-8B-Instruct",
        baseline_pool=(
            "102_qwen35_4b_no_rag_no_think_extractive_mixed_dev",
            "103_qwen35_4b_no_rag_think_extractive_mixed_dev",
        ),
        candidate_pool=(
            "104_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_mixed_dev",
            "105_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_mixed_dev",
        ),
    ),
    # Basque: Llama baseline, Latxa judge.
    MixedAgenticTask(
        run_id="178_mixed_agentic_eu_llama_baseline_latxa_judge_sns1064_dev",
        dev_set="SNS1064 EU dev",
        language="eu",
        baseline_model="Llama baseline",
        judge_model_label="Latxa Agentic Reasoner",
        judge_model="HiTZ/Latxa-Llama-3.1-8B-Instruct",
        baseline_pool=("106_llama31_8b_no_rag_extractive_sns1064_eu_dev",),
        candidate_pool=tuple(f"{idx}_llama31_8b_{suffix}" for idx, suffix in (
            (107, "rag_e5_topk1_extractive_sns1064_eu_dev"),
            (108, "rag_e5_topk3_extractive_sns1064_eu_dev"),
            (109, "rag_e5_topk5_extractive_sns1064_eu_dev"),
            (110, "rag_e5_rerank1_extractive_sns1064_eu_dev"),
            (111, "rag_e5_rerank3_extractive_sns1064_eu_dev"),
            (112, "rag_e5_rerank5_extractive_sns1064_eu_dev"),
            (113, "3shot_no_rag_extractive_sns1064_eu_dev"),
            (114, "rag_3shot_e5_rerank5_extractive_sns1064_eu_dev"),
            (115, "rag_cross_domain_e5_rerank5_extractive_sns1064_eu_dev"),
            (116, "rag_mixed_e5_rerank5_extractive_sns1064_eu_dev"),
        )),
        trust_remote_code=True,
    ),
    MixedAgenticTask(
        run_id="179_mixed_agentic_eu_llama_baseline_latxa_judge_casimedicos_dev",
        dev_set="CasiMedicos EU dev",
        language="eu",
        baseline_model="Llama baseline",
        judge_model_label="Latxa Agentic Reasoner",
        judge_model="HiTZ/Latxa-Llama-3.1-8B-Instruct",
        baseline_pool=("128_llama31_8b_no_rag_extractive_casimedicos_eu_dev",),
        candidate_pool=tuple(f"{idx}_llama31_8b_{suffix}" for idx, suffix in (
            (129, "rag_e5_topk1_extractive_casimedicos_eu_dev"),
            (130, "rag_e5_topk3_extractive_casimedicos_eu_dev"),
            (131, "rag_e5_topk5_extractive_casimedicos_eu_dev"),
            (132, "rag_e5_rerank1_extractive_casimedicos_eu_dev"),
            (133, "rag_e5_rerank3_extractive_casimedicos_eu_dev"),
            (134, "rag_e5_rerank5_extractive_casimedicos_eu_dev"),
            (135, "3shot_no_rag_extractive_casimedicos_eu_dev"),
            (136, "rag_3shot_e5_rerank5_extractive_casimedicos_eu_dev"),
            (137, "rag_cross_domain_e5_rerank5_extractive_casimedicos_eu_dev"),
            (138, "rag_mixed_e5_rerank5_extractive_casimedicos_eu_dev"),
        )),
        trust_remote_code=True,
    ),
    MixedAgenticTask(
        run_id="180_mixed_agentic_eu_llama_baseline_latxa_judge_mixed_dev",
        dev_set="SNS1064+CasiMedicos EU dev",
        language="eu",
        baseline_model="Llama baseline",
        judge_model_label="Latxa Agentic Reasoner",
        judge_model="HiTZ/Latxa-Llama-3.1-8B-Instruct",
        baseline_pool=("150_llama31_8b_no_rag_extractive_mixed_eu_dev",),
        candidate_pool=tuple(f"{idx}_llama31_8b_{suffix}" for idx, suffix in (
            (151, "rag_e5_topk1_extractive_mixed_eu_dev"),
            (152, "rag_e5_topk3_extractive_mixed_eu_dev"),
            (153, "rag_e5_topk5_extractive_mixed_eu_dev"),
            (154, "rag_e5_rerank1_extractive_mixed_eu_dev"),
            (155, "rag_e5_rerank3_extractive_mixed_eu_dev"),
            (156, "rag_e5_rerank5_extractive_mixed_eu_dev"),
            (157, "3shot_no_rag_extractive_mixed_eu_dev"),
            (158, "rag_3shot_e5_rerank5_extractive_mixed_eu_dev"),
            (159, "rag_sns1064_e5_rerank5_extractive_mixed_eu_dev"),
            (160, "rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev"),
        )),
        trust_remote_code=True,
    ),
    # Basque: Latxa baseline, Qwen judge.
    MixedAgenticTask(
        run_id="181_mixed_agentic_eu_latxa_baseline_qwen_judge_sns1064_dev",
        dev_set="SNS1064 EU dev",
        language="eu",
        baseline_model="Latxa baseline",
        judge_model_label="Qwen Agentic Reasoner",
        judge_model="Qwen/Qwen3.5-4B",
        baseline_pool=("117_latxa_llama31_8b_no_rag_extractive_sns1064_eu_dev",),
        candidate_pool=tuple(f"{idx}_latxa_llama31_8b_{suffix}" for idx, suffix in (
            (118, "rag_e5_topk1_extractive_sns1064_eu_dev"),
            (119, "rag_e5_topk3_extractive_sns1064_eu_dev"),
            (120, "rag_e5_topk5_extractive_sns1064_eu_dev"),
            (121, "rag_e5_rerank1_extractive_sns1064_eu_dev"),
            (122, "rag_e5_rerank3_extractive_sns1064_eu_dev"),
            (123, "rag_e5_rerank5_extractive_sns1064_eu_dev"),
            (124, "3shot_no_rag_extractive_sns1064_eu_dev"),
            (125, "rag_3shot_e5_rerank5_extractive_sns1064_eu_dev"),
            (126, "rag_cross_domain_e5_rerank5_extractive_sns1064_eu_dev"),
            (127, "rag_mixed_e5_rerank5_extractive_sns1064_eu_dev"),
        )),
        trust_remote_code=True,
    ),
    MixedAgenticTask(
        run_id="182_mixed_agentic_eu_latxa_baseline_qwen_judge_casimedicos_dev",
        dev_set="CasiMedicos EU dev",
        language="eu",
        baseline_model="Latxa baseline",
        judge_model_label="Qwen Agentic Reasoner",
        judge_model="Qwen/Qwen3.5-4B",
        baseline_pool=("139_latxa_llama31_8b_no_rag_extractive_casimedicos_eu_dev",),
        candidate_pool=tuple(f"{idx}_latxa_llama31_8b_{suffix}" for idx, suffix in (
            (140, "rag_e5_topk1_extractive_casimedicos_eu_dev"),
            (141, "rag_e5_topk3_extractive_casimedicos_eu_dev"),
            (142, "rag_e5_topk5_extractive_casimedicos_eu_dev"),
            (143, "rag_e5_rerank1_extractive_casimedicos_eu_dev"),
            (144, "rag_e5_rerank3_extractive_casimedicos_eu_dev"),
            (145, "rag_e5_rerank5_extractive_casimedicos_eu_dev"),
            (146, "3shot_no_rag_extractive_casimedicos_eu_dev"),
            (147, "rag_3shot_e5_rerank5_extractive_casimedicos_eu_dev"),
            (148, "rag_cross_domain_e5_rerank5_extractive_casimedicos_eu_dev"),
            (149, "rag_mixed_e5_rerank5_extractive_casimedicos_eu_dev"),
        )),
        trust_remote_code=True,
    ),
    MixedAgenticTask(
        run_id="183_mixed_agentic_eu_latxa_baseline_qwen_judge_mixed_dev",
        dev_set="SNS1064+CasiMedicos EU dev",
        language="eu",
        baseline_model="Latxa baseline",
        judge_model_label="Qwen Agentic Reasoner",
        judge_model="Qwen/Qwen3.5-4B",
        baseline_pool=("161_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev",),
        candidate_pool=tuple(f"{idx}_latxa_llama31_8b_{suffix}" for idx, suffix in (
            (162, "rag_e5_topk1_extractive_mixed_eu_dev"),
            (163, "rag_e5_topk3_extractive_mixed_eu_dev"),
            (164, "rag_e5_topk5_extractive_mixed_eu_dev"),
            (165, "rag_e5_rerank1_extractive_mixed_eu_dev"),
            (166, "rag_e5_rerank3_extractive_mixed_eu_dev"),
            (167, "rag_e5_rerank5_extractive_mixed_eu_dev"),
            (168, "3shot_no_rag_extractive_mixed_eu_dev"),
            (169, "rag_3shot_e5_rerank5_extractive_mixed_eu_dev"),
            (170, "rag_sns1064_e5_rerank5_extractive_mixed_eu_dev"),
            (171, "rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev"),
        )),
        trust_remote_code=True,
    ),
)


def metrics_path(run_id: str) -> Path:
    return METRICS_DIR / f"{run_id}.json"


def predictions_path(run_id: str) -> Path:
    return ROOT / "experiments" / "runs" / run_id / "predictions.jsonl"


def no_sf_summary(summary: dict) -> dict:
    return summary.get("before_feedback") or summary


def overall_bert(run_id: str) -> float:
    path = metrics_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return float(no_sf_summary(summary)["overall"]["bertscore_f1"])


def select_best(run_ids: tuple[str, ...]) -> str:
    return max(run_ids, key=overall_bert)


def task_by_index(index: int) -> MixedAgenticTask:
    try:
        return TASKS[index]
    except IndexError as exc:
        raise ValueError(f"Invalid mixed agentic task index {index}; expected 0-{len(TASKS) - 1}.") from exc
