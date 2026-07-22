#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import read_jsonl, write_jsonl  # noqa: E402
from medical_rag_thesis.generation import (  # noqa: E402
    generate_one,
    load_generation_model,
    load_vllm_model,
    make_prompt,
    make_self_feedback_prompt,
    strip_thinking_text,
)
from medical_rag_thesis.prompts import (  # noqa: E402
    MINISTRAL_REASONING_SYSTEM_PROMPT,
    PROMPT_STYLES,
    SYSTEM_PROMPT_ES,
    SYSTEM_PROMPTS,
    build_self_feedback_prompt,
    build_user_prompt,
    query_text,
)
from medical_rag_thesis.reasoning import parse_pipeline_answer  # noqa: E402
from medical_rag_thesis.retrieval import EmbeddingRetriever, LeakLoggingEmbeddingRetriever  # noqa: E402
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
    # Evidence-only experiment: used only when the retrieval index was built by
    # indexing solely the evidence column of the corpus, not the full record.
    parser.add_argument(
        "--log-retrieval-leak", action="store_true",
        help="Evidence-only runs only (see clone_configs_evidence_only.py): use "
        "LeakLoggingEmbeddingRetriever instead of the plain EmbeddingRetriever. "
        "Behaviour is identical (same top-k passages returned), but for every "
        "query it also records whether the query's own gold document would "
        "have appeared in the naive top-(k+1) absent self-retrieval exclusion, "
        "written to <output's parent>/retrieval_leak_log.json alongside "
        "predictions.jsonl.",
    )
    parser.add_argument("--reranker-model", default="", help="Optional CrossEncoder model for reranking retrieved documents.")
    parser.add_argument("--reranker-top-k", type=int, default=0, help="Keep this many documents after reranking.")
    parser.add_argument("--reranker-device", default="cpu", help="Device for the CrossEncoder reranker.")
    parser.add_argument("--language", default="es", choices=["es", "eu"], help="Prompt language (es=Spanish, eu=Basque).")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="Generation engine. vLLM batches every record's prompt into one "
        "engine call per pass (initial, then feedback) instead of the "
        "transformers backend's one-call-per-record loop.",
    )
    parser.add_argument("--max-model-len", type=int, default=16384, help="vLLM backend only.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM backend only.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM backend only.")
    parser.add_argument(
        "--reasoning-parser", default=None,
        help="vLLM backend only. Engine-level reasoning-trace parser name (e.g. "
        "'mistral' for Ministral Reasoning's [THINK]...[/THINK] convention, "
        "'qwen3' for Qwen3.5 think mode), so the trace is stripped from the "
        "returned text before it reaches parse_answer_sections instead of "
        "leaking into the parsed short_answer/evidence fields. None disables it.",
    )
    parser.add_argument(
        "--repetition-detection-max-pattern", type=int, default=0,
        help="vLLM backend only. Stop generation early once a repeating token "
        "pattern up to this size is detected, instead of letting a degenerate "
        "loop run to max_new_tokens and be truncated there. 0 disables it.",
    )
    parser.add_argument("--repetition-detection-min-pattern", type=int, default=1, help="vLLM backend only.")
    parser.add_argument("--repetition-detection-min-count", type=int, default=8, help="vLLM backend only.")
    parser.add_argument(
        "--max-truncation-retries", type=int, default=0,
        help="vLLM backend only. Regenerate any record whose finish_reason is "
        "'length' or 'repetition' (i.e. it was cut off, not a natural EOS "
        "stop), up to this many extra attempts, each with an advanced seed so "
        "a retry is a genuinely different sampling draw rather than a replay "
        "of the same truncated generation.",
    )
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


def uses_ministral_reasoning_prompt(model_name: str) -> bool:
    return "ministral" in model_name.lower() and "reasoning" in model_name.lower()


def resolve_system_prompt(language: str, model_name: str) -> str:
    base = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ES)
    if uses_ministral_reasoning_prompt(model_name):
        return f"{base}\n\n{MINISTRAL_REASONING_SYSTEM_PROMPT}"
    return base


def dry_prompt(
    record: dict[str, Any],
    examples: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    no_think: bool,
    prompt_style: str,
    language: str = "es",
    model_name: str = "",
) -> str:
    system_prompt = resolve_system_prompt(language, model_name)
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
    model_name: str = "",
) -> str:
    system_prompt = resolve_system_prompt(language, model_name)
    answer_label = "Erantzuna" if language == "eu" else "Respuesta"
    user_prompt = build_self_feedback_prompt(
        record,
        answer=answer,
        documents=docs,
        prompt_style=prompt_style,
        language=language,
    )
    return f"{system_prompt}\n\n{user_prompt}\n\n{answer_label}:"


INCOMPLETE_FINISH_REASONS = ("length", "repetition")


def vllm_generate_with_retry(
    llm: Any,
    prompts: list[str],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_truncation_retries: int,
    repetition_detection: Optional[dict[str, int]],
) -> tuple[list[str], list[Optional[str]]]:
    """One batched vLLM call, then up to max_truncation_retries more batched
    calls covering only the records still truncated after the previous
    attempt (finish_reason "length" or "repetition"), each with a freshly
    advanced seed so a retry is a genuinely different sampling draw rather
    than a replay of the same truncated generation (a fixed seed would
    reproduce byte-identical output and the retry would change nothing).

    Returns (texts, finish_reasons), both indexed like `prompts`."""
    from vllm import SamplingParams
    from vllm.sampling_params import RepetitionDetectionParams

    texts: list[Optional[str]] = [None] * len(prompts)
    finish_reasons: list[Optional[str]] = [None] * len(prompts)
    pending_indices = list(range(len(prompts)))
    attempt_seed = seed

    for attempt in range(max_truncation_retries + 1):
        if not pending_indices:
            break
        sp_kwargs: dict[str, Any] = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=attempt_seed,
        )
        if repetition_detection:
            sp_kwargs["repetition_detection"] = RepetitionDetectionParams(**repetition_detection)
        sp = SamplingParams(**sp_kwargs)

        outputs = llm.generate([prompts[i] for i in pending_indices], sp)
        still_pending = []
        for i, output in zip(pending_indices, outputs):
            completion = output.outputs[0]
            texts[i] = strip_thinking_text(completion.text)
            finish_reasons[i] = completion.finish_reason
            if completion.finish_reason in INCOMPLETE_FINISH_REASONS and attempt < max_truncation_retries:
                still_pending.append(i)
        pending_indices = still_pending
        # Offset by attempt index (not a fixed +1) so a third attempt draws a
        # third distinct stream rather than repeating the second attempt's.
        attempt_seed = seed + attempt + 1

    return [t or "" for t in texts], finish_reasons


def run(args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    rng = random.Random(args.seed)
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    few_shot_pool = read_jsonl(args.few_shot_file) if args.few_shot_file else []
    # Evidence-only experiment: LeakLoggingEmbeddingRetriever is only selected
    # for indices built by indexing solely the evidence column of the corpus.
    retriever_cls = LeakLoggingEmbeddingRetriever if args.log_retrieval_leak else EmbeddingRetriever
    retriever = retriever_cls(args.retrieval_index) if args.retrieval_index else None
    reranker = None
    if args.reranker_model and not args.dry_run:
        reranker = load_reranker(args.reranker_model, args.reranker_device)

    tokenizer = model = llm = None
    model_load_seconds = 0.0
    use_vllm = args.backend == "vllm" and not args.dry_run
    if not args.dry_run:
        from transformers import AutoTokenizer

        model_load_started = time.perf_counter()
        if use_vllm:
            # fix_mistral_regex=True: transformers loads Mistral-family tokenizers
            # (Ministral included) with a regex bug by default that silently
            # mis-splits accented/multibyte text into a different, wrong token
            # sequence -- transformers itself warns about this on load. It's a
            # no-op for non-Mistral tokenizers, so passed unconditionally. This
            # only affects THIS tokenizer's own token counts (input_tokens/
            # output_tokens, the cost columns in the result tables) -- the actual
            # generation prompt is sent to vLLM as a raw string and tokenized by
            # vLLM's own (correctly auto-detected MistralTokenizer) engine
            # tokenizer, so generation itself was never affected by this bug.
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=args.trust_remote_code, fix_mistral_regex=True,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            llm = load_vllm_model(
                args.model,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                reasoning_parser=args.reasoning_parser,
            )
        else:
            tokenizer, model = load_generation_model(
                args.model,
                device_map=args.device_map,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )
        model_load_seconds = time.perf_counter() - model_load_started
    native_thinking = uses_native_qwen_thinking(args.model)

    # Pass 1: retrieval, few-shot selection, and prompt construction for every
    # record. This is unchanged regardless of backend -- vLLM only changes how
    # the *generation* calls below are batched, not how prompts are built.
    prepared: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, start=1):
        retrieval_query = query_text(record, language=args.language)
        retrieval_docs: list[dict[str, Any]] = []
        retrieval_candidate_ids: list[str] = []
        retrieval_seconds = 0.0
        rerank_seconds = 0.0
        if retriever and args.retrieval_top_k > 0:
            retrieval_started = time.perf_counter()
            retrieval_docs = retriever.query(retrieval_query, top_k=args.retrieval_top_k, exclude_id=record.get("id"))
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

        few_shot_started = time.perf_counter()
        if args.few_shot_k > 0 and args.few_shot_mode == "retrieval":
            if not retriever:
                raise ValueError("--few-shot-mode retrieval requires --retrieval-index")
            examples = retriever.query(retrieval_query, top_k=args.few_shot_k, exclude_id=record.get("id"))
        else:
            examples = sample_random_examples(rng, few_shot_pool, record, args.few_shot_k)
        few_shot_seconds = time.perf_counter() - few_shot_started

        no_think = not args.think
        prompt_started = time.perf_counter()
        if args.dry_run:
            prompt = dry_prompt(
                record, examples, retrieval_docs, no_think, args.prompt_style, args.language,
                model_name=args.model,
            )
        else:
            assert tokenizer is not None
            prompt = make_prompt(
                tokenizer,
                record,
                examples=examples,
                documents=retrieval_docs,
                no_think=no_think,
                prompt_style=args.prompt_style,
                language=args.language,
                system_prompt=resolve_system_prompt(args.language, args.model),
                enable_thinking=args.think if native_thinking else None,
            )
        prompt_seconds = time.perf_counter() - prompt_started
        input_tokens = len(tokenizer(prompt)["input_ids"]) if tokenizer is not None else None

        prepared.append({
            "ordinal": ordinal,
            "record": record,
            "examples": examples,
            "retrieval_docs": retrieval_docs,
            "retrieval_candidate_ids": retrieval_candidate_ids,
            "prompt": prompt,
            "timing": {
                "retrieval_seconds": retrieval_seconds,
                "rerank_seconds": rerank_seconds,
                "few_shot_seconds": few_shot_seconds,
                "prompt_seconds": prompt_seconds,
            },
            "input_tokens": input_tokens,
        })

    # Pass 2: generation. dry-run writes empty predictions; vLLM batches every
    # record's prompt into one engine call per pass; transformers stays a
    # sequential one-call-per-record loop (generate_one), as before.
    if args.dry_run:
        initial_predictions = [""] * len(prepared)
        for item in prepared:
            item["timing"]["generation_seconds"] = 0.0
    elif use_vllm:
        assert llm is not None
        repetition_detection = (
            {
                "max_pattern_size": args.repetition_detection_max_pattern,
                "min_pattern_size": args.repetition_detection_min_pattern,
                "min_count": args.repetition_detection_min_count,
            }
            if args.repetition_detection_max_pattern
            else None
        )
        gen_started = time.perf_counter()
        initial_predictions, initial_finish_reasons = vllm_generate_with_retry(
            llm,
            [item["prompt"] for item in prepared],
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_truncation_retries=args.max_truncation_retries,
            repetition_detection=repetition_detection,
        )
        # vLLM batches the whole pass into one engine call, so there is no true
        # per-record wall time; the batch's total is divided evenly across its
        # records to keep mean_generation_seconds comparable to the
        # transformers backend's per-record timing.
        per_record_seconds = (time.perf_counter() - gen_started) / len(prepared) if prepared else 0.0
        for item, reason in zip(prepared, initial_finish_reasons):
            item["timing"]["generation_seconds"] = per_record_seconds
            item["initial_finish_reason"] = reason
    else:
        assert tokenizer is not None and model is not None
        initial_predictions = []
        for item in prepared:
            generation_started = time.perf_counter()
            initial_predictions.append(
                generate_one(
                    tokenizer,
                    model,
                    item["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_thinking=args.think if native_thinking else None,
                )
            )
            item["timing"]["generation_seconds"] = time.perf_counter() - generation_started

    # Pass 3: self-feedback, built from each record's own initial prediction.
    feedback_prompts: list[Optional[str]] = [None] * len(prepared)
    if args.self_feedback:
        for item, initial_prediction in zip(prepared, initial_predictions):
            record = item["record"]
            if args.dry_run:
                feedback_prompts[item["ordinal"] - 1] = dry_self_feedback_prompt(
                    record, item["retrieval_docs"], initial_prediction, args.prompt_style, args.language,
                    model_name=args.model,
                )
            else:
                assert tokenizer is not None
                feedback_prompts[item["ordinal"] - 1] = make_self_feedback_prompt(
                    tokenizer,
                    record,
                    answer=initial_prediction,
                    documents=item["retrieval_docs"],
                    prompt_style=args.prompt_style,
                    language=args.language,
                    system_prompt=resolve_system_prompt(args.language, args.model),
                    enable_thinking=args.think if native_thinking else None,
                )

        if args.dry_run:
            predictions = list(initial_predictions)
            for item in prepared:
                item["timing"]["feedback_generation_seconds"] = 0.0
        elif use_vllm:
            assert llm is not None
            repetition_detection = (
                {
                    "max_pattern_size": args.repetition_detection_max_pattern,
                    "min_pattern_size": args.repetition_detection_min_pattern,
                    "min_count": args.repetition_detection_min_count,
                }
                if args.repetition_detection_max_pattern
                else None
            )
            fb_started = time.perf_counter()
            predictions, feedback_finish_reasons = vllm_generate_with_retry(
                llm,
                feedback_prompts,
                max_tokens=args.feedback_max_new_tokens or args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + 1,
                max_truncation_retries=args.max_truncation_retries,
                repetition_detection=repetition_detection,
            )
            fb_seconds = (time.perf_counter() - fb_started) / len(prepared) if prepared else 0.0
            for item, reason in zip(prepared, feedback_finish_reasons):
                item["timing"]["feedback_generation_seconds"] = fb_seconds
                item["feedback_finish_reason"] = reason
        else:
            assert tokenizer is not None and model is not None
            predictions = []
            for item, feedback_prompt in zip(prepared, feedback_prompts):
                feedback_started = time.perf_counter()
                predictions.append(
                    generate_one(
                        tokenizer,
                        model,
                        feedback_prompt,
                        max_new_tokens=args.feedback_max_new_tokens or args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        enable_thinking=args.think if native_thinking else None,
                    )
                )
                item["timing"]["feedback_generation_seconds"] = time.perf_counter() - feedback_started
    else:
        predictions = list(initial_predictions)
        for item in prepared:
            item["timing"]["feedback_generation_seconds"] = 0.0

    outputs = []
    for item, initial_prediction, prediction, feedback_prompt in zip(
        prepared, initial_predictions, predictions, feedback_prompts
    ):
        record = item["record"]
        examples = item["examples"]
        retrieval_docs = item["retrieval_docs"]
        timing = item["timing"]
        # For vLLM, generation/feedback seconds are the batch's wall time divided
        # evenly across its records (there is no true per-record time inside a
        # batched call), so example_seconds here is likewise a components sum
        # rather than a single wall-clock bracket, for every backend uniformly.
        example_seconds = (
            timing["retrieval_seconds"] + timing["rerank_seconds"] + timing["few_shot_seconds"]
            + timing["prompt_seconds"] + timing["generation_seconds"] + timing["feedback_generation_seconds"]
        )
        feedback_input_tokens = len(tokenizer(feedback_prompt)["input_ids"]) if (feedback_prompt and tokenizer is not None) else None
        initial_output_tokens = len(tokenizer(initial_prediction)["input_ids"]) if (initial_prediction and tokenizer is not None) else (0 if not args.dry_run else None)
        output_tokens = len(tokenizer(prediction)["input_ids"]) if (prediction and tokenizer is not None) else (0 if not args.dry_run else None)

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
            # parse_pipeline_answer (not the plain parse_answer_sections) anchors
            # on the LAST "Respuesta corta"/"Evidencia" occurrence -- required for
            # think-mode output, which routinely rehearses those labels mid-reasoning
            # before the real final answer (see scripts/refix_think_mode_parsing.py
            # for the incident this was originally caught from: a no-think call site
            # here was left on the plain parser after think-mode support was added,
            # silently corrupting every think-mode run's short_answer/evidence until
            # re-parsed from raw text after the fact). It degrades to identical
            # behavior on no-think output (single occurrence), so it is used
            # unconditionally rather than branching on args.think.
            "parsed_initial_prediction": parse_pipeline_answer(initial_prediction) if initial_prediction else {},
            "prediction_text": prediction,
            "parsed_prediction": parse_pipeline_answer(prediction) if prediction else {},
            "few_shot_ids": [example.get("doc_id") or example.get("id") for example in examples],
            "retrieval_docs": retrieval_docs,
            "retrieval_candidate_ids": item["retrieval_candidate_ids"],
            "timing": {
                "retrieval_seconds": timing["retrieval_seconds"],
                "rerank_seconds": timing["rerank_seconds"],
                "few_shot_seconds": timing["few_shot_seconds"],
                "prompt_seconds": timing["prompt_seconds"],
                "generation_seconds": timing["generation_seconds"],
                "feedback_generation_seconds": timing["feedback_generation_seconds"],
                "example_seconds": example_seconds,
            },
            "token_counts": {
                "input_tokens": item["input_tokens"],
                "feedback_input_tokens": feedback_input_tokens,
                "initial_output_tokens": initial_output_tokens,
                "output_tokens": output_tokens,
            },
            "truncation": {
                "initial_finish_reason": item.get("initial_finish_reason"),
                "initial_truncated": item.get("initial_finish_reason") in INCOMPLETE_FINISH_REASONS,
                "feedback_finish_reason": item.get("feedback_finish_reason"),
                "feedback_truncated": item.get("feedback_finish_reason") in INCOMPLETE_FINISH_REASONS,
            },
        }
        if args.save_prompts or args.dry_run:
            output["prompt"] = item["prompt"]
            if feedback_prompt is not None:
                output["feedback_prompt"] = feedback_prompt
        outputs.append(output)
        print(f"[{item['ordinal']}/{len(records)}] {record.get('id', '')}", flush=True)

    write_jsonl(outputs, args.output)
    metadata_path = Path(args.output).with_suffix(".meta.json")
    total_run_seconds = time.perf_counter() - run_started
    generation_times = [record["timing"]["generation_seconds"] for record in outputs]
    example_times = [record["timing"]["example_seconds"] for record in outputs]
    truncation_counts = {
        "num_initial_truncated": sum(1 for r in outputs if r["truncation"]["initial_truncated"]),
        "num_feedback_truncated": sum(1 for r in outputs if r["truncation"]["feedback_truncated"]),
        "num_records": len(outputs),
    }
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
        "max_truncation_retries": args.max_truncation_retries,
        "truncation_counts": truncation_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Evidence-only experiment: sidecar leak-log output, only produced for
    # indices built by indexing solely the evidence column of the corpus.
    if args.log_retrieval_leak and retriever is not None:
        leak_log = retriever.leak_log
        num_present = sum(1 for entry in leak_log if entry["excluded_present"])
        leak_summary = {
            "num_queries": len(leak_log),
            "num_gold_present_in_naive_topk_plus_1": num_present,
            "leak_rate": num_present / len(leak_log) if leak_log else None,
            "entries": leak_log,
        }
        leak_path = Path(args.output).parent / "retrieval_leak_log.json"
        leak_path.write_text(json.dumps(leak_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"retrieval leak log: {num_present}/{len(leak_log)} queries "
              f"({leak_summary['leak_rate']:.1%})" if leak_log else "retrieval leak log: no queries"
              , flush=True)


if __name__ == "__main__":
    parsed_args = parse_args()
    run_dir = Path(parsed_args.output).parent
    run_with_logs(run_dir / "run.log", run_dir / "run.err", lambda: run(parsed_args))
