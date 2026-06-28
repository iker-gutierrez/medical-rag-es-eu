#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import read_jsonl, write_jsonl  # noqa: E402
from medical_rag_thesis.generation import (  # noqa: E402
    build_chat_prompt,
    generate_one,
    load_generation_model,
)
from medical_rag_thesis.prompts import (  # noqa: E402
    SYSTEM_PROMPT_ES,
    SYSTEM_PROMPTS,
    format_context_text,
    format_options,
    format_question,
    parse_answer_sections,
)
from medical_rag_thesis.run_logging import run_with_logs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agentic verifier over baseline and best-system predictions."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--language", default="es", choices=["es", "eu"])
    parser.add_argument("--think", action="store_true", help="Enable native thinking for models that support it.")
    parser.add_argument(
        "--source-answer-key",
        choices=("initial", "final"),
        default="initial",
        help="Use no-self-feedback source answers by default.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    return parser.parse_args()


def metadata_path(predictions_path: Path) -> Path:
    return predictions_path.with_suffix(".meta.json")


def load_metadata(predictions_path: Path) -> dict[str, Any]:
    path = metadata_path(predictions_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        record_id = str(record.get("id") or "")
        if record_id:
            indexed[record_id] = record
    return indexed


def source_text(record: Mapping[str, Any], answer_key: str) -> str:
    if answer_key == "initial":
        return str(record.get("initial_prediction_text") or record.get("prediction_text") or "").strip()
    return str(record.get("prediction_text") or record.get("initial_prediction_text") or "").strip()


def format_candidate_answer(
    label: str,
    record: Mapping[str, Any],
    answer_key: str,
    language: str = "es",
) -> str:
    text = source_text(record, answer_key)
    parsed_key = "parsed_initial_prediction" if answer_key == "initial" else "parsed_prediction"
    parsed = record.get(parsed_key) or parse_answer_sections(text)
    short_answer = str(parsed.get("short_answer") or "").strip()
    evidence = str(parsed.get("evidence") or "").strip()
    answer_label = "Erantzun laburra" if language == "eu" else "Respuesta corta"
    evidence_label = "Ebidentzia" if language == "eu" else "Evidencia"
    if short_answer or evidence:
        return "\n".join(
            part
            for part in [
                f"{label}",
                f"{answer_label}: {short_answer}" if short_answer else "",
                f"{evidence_label}: {evidence}" if evidence else "",
            ]
            if part
        )
    return f"{label}\n{text}"


def uses_native_qwen_thinking(model_name: str) -> bool:
    normalized = model_name.lower()
    return "qwen/qwen3" in normalized or "qwen3" in normalized


def build_agentic_prompt(
    *,
    record: Mapping[str, Any],
    baseline_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    answer_key: str,
    language: str = "es",
) -> str:
    question_label = "Galdera" if language == "eu" else "Pregunta"
    options_label = "Aukerak" if language == "eu" else "Opciones"
    context_label = "Errekuperatutako testuingurua" if language == "eu" else "Contexto recuperado"
    baseline_label = "A erantzuna (LLM hutsa):" if language == "eu" else "Respuesta A (LLM-only):"
    candidate_label = (
        "B erantzuna (baseline ez den konfigurazio onena):"
        if language == "eu"
        else "Respuesta B (mejor configuración):"
    )
    final_label = "Azken erantzuna:" if language == "eu" else "Respuesta final:"

    question = format_question(record, language=language).removeprefix(f"{question_label}: ").strip()
    options = format_options(record)
    context_text = format_context_text(candidate_record.get("retrieval_docs") or [])

    if language == "eu":
        sections = [
            "Egiaztatzaile klinikoa zara.",
            "Zure zeregina bi erantzun hautagaitatik azken erantzun bat sortzea da:",
            "- A erantzuna: errekuperaziorik gabeko LLM baten irteera.",
            "- B erantzuna: baseline ez den konfigurazio onenaren irteera.",
            "Arauak:",
            "- Alderatu bi erantzunak eta gorde galderari erantzuten dion informazioa soilik.",
            "- Errekuperatutako testuingurua badago, lehenetsi testuinguru horretan oinarritutako informazioa.",
            "- Erantzunak kontraesanean badaude, aukeratu hobeto justifikatutako eta zehatzagoa den aukera.",
            "- Ez asmatu datu zehatzik.",
            "- Erantzun euskaraz.",
            "- Erantzun SOILIK bi eremu hauekin:\n\n"
            "Erantzun laburra:\n"
            "(erantzun laburra)\n\n"
            "Ebidentzia:\n"
            "(justifikazioa)",
        ]
    else:
        sections = [
            "Eres un verificador clínico.",
            "Tu tarea es crear una respuesta final a partir de dos respuestas candidatas:",
            "- Respuesta A: salida de un LLM sin recuperación.",
            "- Respuesta B: salida de la mejor configuración no-baseline.",
            "Reglas:",
            "- Compara las dos respuestas y conserva solo la información que responda a la pregunta.",
            "- Si hay contexto recuperado, prioriza la información apoyada por ese contexto.",
            "- Si las respuestas se contradicen, elige la opción mejor justificada y más específica.",
            "- No inventes datos concretos.",
            "- Responde en español.",
            "- Responde SOLO con estos dos campos:\n\n"
            "Respuesta corta:\n"
            "(respuesta breve)\n\n"
            "Evidencia:\n"
            "(justificación)",
        ]
    if context_text:
        context_detail = (
            "baseline ez den konfigurazio onenarena"
            if language == "eu"
            else "de la mejor configuración"
        )
        sections.append(f"{context_label} {context_detail}:\n" + context_text)
    sections.append(f"{question_label}:\n" + question)
    if options:
        sections.append(f"{options_label}:\n" + options)
    sections.append(format_candidate_answer(baseline_label, baseline_record, answer_key, language=language))
    sections.append(format_candidate_answer(candidate_label, candidate_record, answer_key, language=language))
    sections.append(final_label)
    return "\n\n".join(sections)


def no_sf_timing(record: Mapping[str, Any]) -> dict[str, float]:
    timing = record.get("timing") or {}

    def get(name: str) -> float:
        value = timing.get(name)
        return float(value) if value is not None else 0.0

    return {
        "retrieval_seconds": get("retrieval_seconds"),
        "rerank_seconds": get("rerank_seconds"),
        "few_shot_seconds": get("few_shot_seconds"),
        "prompt_seconds": get("prompt_seconds"),
        "generation_seconds": get("generation_seconds"),
        "feedback_generation_seconds": 0.0,
        "example_seconds": (
            get("retrieval_seconds")
            + get("rerank_seconds")
            + get("few_shot_seconds")
            + get("prompt_seconds")
            + get("generation_seconds")
        ),
    }


def no_sf_tokens(record: Mapping[str, Any]) -> dict[str, int]:
    counts = record.get("token_counts") or {}

    def get(name: str) -> int:
        value = counts.get(name)
        return int(value) if value is not None else 0

    return {
        "input_tokens": get("input_tokens"),
        "output_tokens": get("initial_output_tokens") or get("output_tokens"),
    }


def add_timing(*items: Mapping[str, float]) -> dict[str, float]:
    fields = (
        "retrieval_seconds",
        "rerank_seconds",
        "few_shot_seconds",
        "prompt_seconds",
        "generation_seconds",
        "feedback_generation_seconds",
        "example_seconds",
    )
    return {field: sum(float(item.get(field) or 0.0) for item in items) for field in fields}


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    baseline_path = resolve_path(args.baseline_predictions)
    candidate_path = resolve_path(args.candidate_predictions)
    output_path = resolve_path(args.output)

    baseline_records = read_jsonl(baseline_path)
    candidate_records = read_jsonl(candidate_path)
    baseline_index = by_id(baseline_records)
    candidate_index = by_id(candidate_records)
    common_ids = [record_id for record_id in baseline_index if record_id in candidate_index]
    if args.limit is not None:
        common_ids = common_ids[: args.limit]
    if not common_ids:
        raise ValueError("No overlapping record ids between baseline and candidate predictions.")

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

    outputs = []
    for ordinal, record_id in enumerate(common_ids, start=1):
        example_started = time.perf_counter()
        baseline_record = baseline_index[record_id]
        candidate_record = candidate_index[record_id]
        prompt_started = time.perf_counter()
        user_prompt = build_agentic_prompt(
            record=candidate_record,
            baseline_record=baseline_record,
            candidate_record=candidate_record,
            answer_key=args.source_answer_key,
            language=args.language,
        )
        system_prompt = SYSTEM_PROMPTS.get(args.language, SYSTEM_PROMPT_ES)
        enable_thinking = args.think if uses_native_qwen_thinking(args.model) else None
        if args.dry_run:
            answer_label = "Erantzuna" if args.language == "eu" else "Respuesta"
            prompt = f"{system_prompt}\n\n{user_prompt}\n\n{answer_label}:"
            prediction = ""
            input_tokens = None
            output_tokens = None
        else:
            assert tokenizer is not None and model is not None
            prompt = build_chat_prompt(
                tokenizer,
                system_prompt,
                user_prompt,
                language=args.language,
                enable_thinking=enable_thinking,
            )
            input_tokens = len(tokenizer(prompt)["input_ids"])
            generation_started = time.perf_counter()
            prediction = generate_one(
                tokenizer,
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=enable_thinking,
            )
            judge_generation_seconds = time.perf_counter() - generation_started
            output_tokens = len(tokenizer(prediction)["input_ids"]) if prediction else 0
        prompt_seconds = time.perf_counter() - prompt_started
        if args.dry_run:
            judge_generation_seconds = 0.0

        baseline_timing = no_sf_timing(baseline_record)
        candidate_timing = no_sf_timing(candidate_record)
        judge_timing = {
            "retrieval_seconds": 0.0,
            "rerank_seconds": 0.0,
            "few_shot_seconds": 0.0,
            "prompt_seconds": prompt_seconds,
            "generation_seconds": judge_generation_seconds,
            "feedback_generation_seconds": 0.0,
            "example_seconds": time.perf_counter() - example_started,
        }
        timing = add_timing(baseline_timing, candidate_timing, judge_timing)

        baseline_tokens = no_sf_tokens(baseline_record)
        candidate_tokens = no_sf_tokens(candidate_record)
        if input_tokens is None:
            total_input_tokens = None
            total_output_tokens = None
        else:
            total_input_tokens = baseline_tokens["input_tokens"] + candidate_tokens["input_tokens"] + input_tokens
            total_output_tokens = baseline_tokens["output_tokens"] + candidate_tokens["output_tokens"] + (output_tokens or 0)

        output = {
            "id": candidate_record.get("id"),
            "experiment_name": args.experiment_name,
            "model": args.model,
            "prompt_style": "agentic_reasoner",
            "rag_condition": "agentic",
            "reasoning_condition": "judge_verifier",
            "language": args.language,
            "native_thinking": enable_thinking,
            "source": candidate_record.get("source"),
            "topic": candidate_record.get("topic"),
            "question": candidate_record.get("question"),
            "subquestion": candidate_record.get("subquestion", ""),
            "reference_short_answer": candidate_record.get("reference_short_answer", ""),
            "reference_evidence": candidate_record.get("reference_evidence", ""),
            "prediction_text": prediction,
            "parsed_prediction": parse_answer_sections(prediction) if prediction else {},
            "agentic_sources": {
                "baseline_run": baseline_record.get("experiment_name"),
                "candidate_run": candidate_record.get("experiment_name"),
                "source_answer_key": args.source_answer_key,
                "baseline_prediction": source_text(baseline_record, args.source_answer_key),
                "candidate_prediction": source_text(candidate_record, args.source_answer_key),
            },
            "retrieval_docs": candidate_record.get("retrieval_docs") or [],
            "retrieval_candidate_ids": candidate_record.get("retrieval_candidate_ids") or [],
            "timing": timing,
            "token_counts": {
                "input_tokens": total_input_tokens,
                "feedback_input_tokens": None,
                "initial_output_tokens": total_output_tokens,
                "output_tokens": total_output_tokens,
            },
            "agentic_component_cost": {
                "baseline_no_sf_timing": baseline_timing,
                "candidate_no_sf_timing": candidate_timing,
                "judge_timing": judge_timing,
                "baseline_no_sf_tokens": baseline_tokens,
                "candidate_no_sf_tokens": candidate_tokens,
                "judge_input_tokens": input_tokens,
                "judge_output_tokens": output_tokens,
            },
        }
        if args.save_prompts or args.dry_run:
            output["prompt"] = prompt
        outputs.append(output)
        print(f"[{ordinal}/{len(common_ids)}] {record_id}", flush=True)

    write_jsonl(outputs, output_path)
    baseline_meta = load_metadata(baseline_path)
    candidate_meta = load_metadata(candidate_path)
    total_run_seconds = time.perf_counter() - started
    metadata = {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "prompt_style": "agentic_reasoner",
        "rag_condition": "agentic",
        "reasoning_condition": "judge_verifier",
        "language": args.language,
        "native_thinking": args.think if uses_native_qwen_thinking(args.model) else None,
        "input": candidate_meta.get("input") or baseline_meta.get("input"),
        "output": str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
        "num_records": len(outputs),
        "dry_run": args.dry_run,
        "model_load_seconds": model_load_seconds,
        "total_run_seconds": total_run_seconds,
        "mean_generation_seconds": (
            sum((record["agentic_component_cost"]["judge_timing"]["generation_seconds"] for record in outputs))
            / len(outputs)
            if outputs
            else 0.0
        ),
        "mean_example_seconds": (
            sum((record["timing"]["example_seconds"] for record in outputs)) / len(outputs) if outputs else 0.0
        ),
        "baseline_predictions": str(baseline_path.relative_to(ROOT) if baseline_path.is_relative_to(ROOT) else baseline_path),
        "candidate_predictions": str(
            candidate_path.relative_to(ROOT) if candidate_path.is_relative_to(ROOT) else candidate_path
        ),
        "baseline_experiment_name": baseline_meta.get("experiment_name"),
        "candidate_experiment_name": candidate_meta.get("experiment_name"),
        "source_answer_key": args.source_answer_key,
        "self_feedback": False,
    }
    metadata_path(output_path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output)
    run_with_logs(output_path.with_suffix(".log"), output_path.with_suffix(".err"), lambda: run(args))


if __name__ == "__main__":
    main()
