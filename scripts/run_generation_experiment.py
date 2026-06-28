#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import read_jsonl, write_jsonl  # noqa: E402
from medical_rag_thesis.generation import (  # noqa: E402
    generate_one,
    load_generation_model,
    make_prompt,
    make_self_feedback_prompt,
)
from medical_rag_thesis.prompts import (  # noqa: E402
    PROMPT_STYLES,
    SYSTEM_PROMPT_ES,
    SYSTEM_PROMPTS,
    build_self_feedback_prompt,
    build_user_prompt,
    parse_answer_sections,
    query_text,
)
from medical_rag_thesis.retrieval import EmbeddingRetriever  # noqa: E402
from medical_rag_thesis.run_logging import run_with_logs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-only, few-shot, or retrieval-augmented generation."
    )
    parser.add_argument("--input", required=True, help="Input split JSONL.")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-style", default="extractive", choices=PROMPT_STYLES)
    parser.add_argument("--think", action="store_true", help="Allow chain-of-thought style prompting.")
    parser.add_argument("--self-feedback", action="store_true", help="Generate an initial answer, then refine it with a self-feedback prompt.")
    parser.add_argument("--feedback-max-new-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without loading a model.")
    parser.add_argument("--save-prompts", action="store_true")

    parser.add_argument("--few-shot-file", help="JSONL pool for few-shot examples.")
    parser.add_argument("--few-shot-k", type=int, default=0)
    parser.add_argument(
        "--few-shot-mode",
        default="random",
        choices=["random", "retrieval"],
        help="How to choose few-shot examples when --few-shot-k > 0.",
    )

    parser.add_argument("--retrieval-index", help="Index directory for context retrieval.")
    parser.add_argument("--retrieval-top-k", type=int, default=0)
    parser.add_argument("--reranker-model", default="", help="Optional CrossEncoder model for reranking retrieved documents.")
    parser.add_argument("--reranker-top-k", type=int, default=0, help="Keep this many documents after reranking.")
    parser.add_argument("--reranker-device", default="cpu", help="Device for the CrossEncoder reranker.")
    parser.add_argument("--language", default="es", choices=["es", "eu"], help="Prompt language (es=Spanish, eu=Basque).")
    return parser.parse_args()


def sample_random_examples(
    rng: random.Random,
    pool: list[dict[str, Any]],
    record: dict[str, Any],
    k: int,
) -> list[dict[str, Any]]:
    candidates = [item for item in pool if item.get("id") != record.get("id")]
    if not candidates or k <= 0:
        return []
    return rng.sample(candidates, k=min(k, len(candidates)))


def load_reranker(model_name: str, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def rerank_documents(
    reranker: Any,
    query: str,
    documents: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not documents or top_k <= 0:
        return documents
    pairs = [(query, str(document.get("text") or "")) for document in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
    reranked = []
    for rank, (document, score) in enumerate(ranked[:top_k], start=1):
        item = dict(document)
        item["pre_rerank_rank"] = item.get("rank")
        item["pre_rerank_score"] = item.get("score")
        item["reranker_score"] = float(score)
        item["rank"] = rank
        reranked.append(item)
    return reranked


def uses_native_qwen_thinking(model_name: str) -> bool:
    normalized = model_name.lower()
    return "qwen/qwen3" in normalized or "qwen3" in normalized


def dry_prompt(
    record: dict[str, Any],
    examples: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    no_think: bool,
    prompt_style: str,
    language: str = "es",
) -> str:
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ES)
    answer_label = "Erantzuna" if language == "eu" else "Respuesta"
    user_prompt = build_user_prompt(
        record,
        examples=examples,
        documents=docs,
        no_think=no_think,
        prompt_style=prompt_style,
        language=language,
    )
    return f"{system_prompt}\n\n{user_prompt}\n\n{answer_label}:"


def dry_self_feedback_prompt(
    record: dict[str, Any],
    docs: list[dict[str, Any]],
    answer: str,
    prompt_style: str,
    language: str = "es",
) -> str:
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ES)
    answer_label = "Erantzuna" if language == "eu" else "Respuesta"
    user_prompt = build_self_feedback_prompt(
        record,
        answer=answer,
        documents=docs,
        prompt_style=prompt_style,
        language=language,
    )
    return f"{system_prompt}\n\n{user_prompt}\n\n{answer_label}:"


def run(args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    rng = random.Random(args.seed)
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    few_shot_pool = read_jsonl(args.few_shot_file) if args.few_shot_file else []
    retriever = EmbeddingRetriever(args.retrieval_index) if args.retrieval_index else None
    reranker = None
    if args.reranker_model and not args.dry_run:
        reranker = load_reranker(args.reranker_model, args.reranker_device)

    tokenizer = model = None
    model_load_seconds = 0.0
    if not args.dry_run:
        model_load_started = time.perf_counter()
        tokenizer, model = load_generation_model(
            args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model_load_seconds = time.perf_counter() - model_load_started
    native_thinking = uses_native_qwen_thinking(args.model)

    outputs = []
    for ordinal, record in enumerate(records, start=1):
        example_started = time.perf_counter()
        retrieval_query = query_text(record, language=args.language)
        retrieval_docs: list[dict[str, Any]] = []
        retrieval_candidate_ids: list[str] = []
        retrieval_seconds = 0.0
        rerank_seconds = 0.0
        if retriever and args.retrieval_top_k > 0:
            retrieval_started = time.perf_counter()
            retrieval_docs = retriever.query(retrieval_query, top_k=args.retrieval_top_k)
            retrieval_seconds = time.perf_counter() - retrieval_started
            retrieval_candidate_ids = [str(doc.get("doc_id") or doc.get("id") or "") for doc in retrieval_docs]
            if args.reranker_model and args.reranker_top_k > 0:
                rerank_started = time.perf_counter()
                if reranker is not None:
                    retrieval_docs = rerank_documents(
                        reranker,
                        retrieval_query,
                        retrieval_docs,
                        top_k=args.reranker_top_k,
                    )
                else:
                    retrieval_docs = retrieval_docs[: args.reranker_top_k]
                rerank_seconds = time.perf_counter() - rerank_started

        few_shot_seconds = 0.0
        few_shot_started = time.perf_counter()
        if args.few_shot_k > 0 and args.few_shot_mode == "retrieval":
            if not retriever:
                raise ValueError("--few-shot-mode retrieval requires --retrieval-index")
            examples = retriever.query(retrieval_query, top_k=args.few_shot_k)
        else:
            examples = sample_random_examples(rng, few_shot_pool, record, args.few_shot_k)
        few_shot_seconds = time.perf_counter() - few_shot_started

        no_think = not args.think
        prompt_started = time.perf_counter()
        if args.dry_run:
            prompt = dry_prompt(record, examples, retrieval_docs, no_think, args.prompt_style, args.language)
            initial_prediction = ""
            prediction = ""
            feedback_prompt = (
                dry_self_feedback_prompt(record, retrieval_docs, initial_prediction, args.prompt_style, args.language)
                if args.self_feedback
                else None
            )
            prompt_seconds = time.perf_counter() - prompt_started
            generation_seconds = 0.0
            feedback_generation_seconds = 0.0
            input_tokens = None
            feedback_input_tokens = None
            output_tokens = None
            initial_output_tokens = None
        else:
            assert tokenizer is not None and model is not None
            prompt = make_prompt(
                tokenizer,
                record,
                examples=examples,
                documents=retrieval_docs,
                no_think=no_think,
                prompt_style=args.prompt_style,
                language=args.language,
                enable_thinking=args.think if native_thinking else None,
            )
            prompt_seconds = time.perf_counter() - prompt_started
            input_tokens = len(tokenizer(prompt)["input_ids"])
            generation_started = time.perf_counter()
            initial_prediction = generate_one(
                tokenizer,
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.think if native_thinking else None,
            )
            generation_seconds = time.perf_counter() - generation_started
            prediction = initial_prediction
            feedback_prompt = None
            feedback_generation_seconds = 0.0
            feedback_input_tokens = None
            initial_output_tokens = len(tokenizer(initial_prediction)["input_ids"]) if initial_prediction else 0
            if args.self_feedback:
                feedback_prompt = make_self_feedback_prompt(
                    tokenizer,
                    record,
                    answer=initial_prediction,
                    documents=retrieval_docs,
                    prompt_style=args.prompt_style,
                    language=args.language,
                    enable_thinking=args.think if native_thinking else None,
                )
                feedback_input_tokens = len(tokenizer(feedback_prompt)["input_ids"])
                feedback_started = time.perf_counter()
                prediction = generate_one(
                    tokenizer,
                    model,
                    feedback_prompt,
                    max_new_tokens=args.feedback_max_new_tokens or args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_thinking=args.think if native_thinking else None,
                )
                feedback_generation_seconds = time.perf_counter() - feedback_started
            output_tokens = len(tokenizer(prediction)["input_ids"]) if prediction else 0

        example_seconds = time.perf_counter() - example_started
        output = {
            "id": record.get("id"),
            "experiment_name": args.experiment_name,
            "model": args.model,
            "prompt_style": args.prompt_style,
            "rag_condition": "rag" if args.retrieval_index and args.retrieval_top_k > 0 else "no_rag",
            "reasoning_condition": "think" if args.think else "no_think",
            "native_thinking": native_thinking,
            "source": record.get("source"),
            "topic": record.get("topic"),
            "question": record.get("question"),
            "subquestion": record.get("subquestion", ""),
            "reference_short_answer": record.get("short_answer", ""),
            "reference_evidence": record.get("evidence", ""),
            "initial_prediction_text": initial_prediction,
            "parsed_initial_prediction": parse_answer_sections(initial_prediction) if initial_prediction else {},
            "prediction_text": prediction,
            "parsed_prediction": parse_answer_sections(prediction) if prediction else {},
            "few_shot_ids": [example.get("doc_id") or example.get("id") for example in examples],
            "retrieval_docs": retrieval_docs,
            "retrieval_candidate_ids": retrieval_candidate_ids,
            "timing": {
                "retrieval_seconds": retrieval_seconds,
                "rerank_seconds": rerank_seconds,
                "few_shot_seconds": few_shot_seconds,
                "prompt_seconds": prompt_seconds,
                "generation_seconds": generation_seconds,
                "feedback_generation_seconds": feedback_generation_seconds,
                "example_seconds": example_seconds,
            },
            "token_counts": {
                "input_tokens": input_tokens,
                "feedback_input_tokens": feedback_input_tokens,
                "initial_output_tokens": initial_output_tokens,
                "output_tokens": output_tokens,
            },
        }
        if args.save_prompts or args.dry_run:
            output["prompt"] = prompt
            if feedback_prompt is not None:
                output["feedback_prompt"] = feedback_prompt
        outputs.append(output)
        print(f"[{ordinal}/{len(records)}] {record.get('id', '')}", flush=True)

    write_jsonl(outputs, args.output)
    metadata_path = Path(args.output).with_suffix(".meta.json")
    total_run_seconds = time.perf_counter() - run_started
    generation_times = [record["timing"]["generation_seconds"] for record in outputs]
    example_times = [record["timing"]["example_seconds"] for record in outputs]
    metadata = {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "prompt_style": args.prompt_style,
        "rag_condition": "rag" if args.retrieval_index and args.retrieval_top_k > 0 else "no_rag",
        "reasoning_condition": "think" if args.think else "no_think",
        "native_thinking": native_thinking,
        "input": args.input,
        "output": args.output,
        "num_records": len(outputs),
        "dry_run": args.dry_run,
        "model_load_seconds": model_load_seconds,
        "total_run_seconds": total_run_seconds,
        "mean_generation_seconds": sum(generation_times) / len(generation_times) if generation_times else 0.0,
        "mean_example_seconds": sum(example_times) / len(example_times) if example_times else 0.0,
        "few_shot_k": args.few_shot_k,
        "few_shot_mode": args.few_shot_mode,
        "retrieval_index": args.retrieval_index,
        "retrieval_top_k": args.retrieval_top_k,
        "reranker_model": args.reranker_model,
        "reranker_top_k": args.reranker_top_k,
        "reranker_device": args.reranker_device,
        "self_feedback": args.self_feedback,
        "feedback_max_new_tokens": args.feedback_max_new_tokens or args.max_new_tokens,
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parsed_args = parse_args()
    run_dir = Path(parsed_args.output).parent
    run_with_logs(run_dir / "run.log", run_dir / "run.err", lambda: run(parsed_args))
