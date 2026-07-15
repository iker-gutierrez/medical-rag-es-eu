#!/usr/bin/env python
"""Single-request (batch-size-1) latency for a chosen configuration.

The dev ablation reports `sec/sample` as a *batched* mean: all dev questions are
served together via vLLM continuous batching, and the total is divided by the
count. That is the right metric for comparing configurations -- it measures
amortized serving cost on identical hardware -- but it is NOT how long one
clinician waits for one answer. Batching overlaps requests, so the batched
per-sample figure understates true single-request latency, often by several times
for reasoning configs that decode long traces.

This script measures the other quantity: the wall-clock time for a *single* request,
end to end (retrieval + rerank + prompt build + generation, and the self-feedback
pass if the config uses it), by issuing one request at a time and timing each. It is
meant to be run ONCE on the finally-chosen configuration, over a small sample -- not
across the grid. Latency does not depend on the seed and is stable over ~20-30
records, so a small sample gives a solid mean without re-running anything.

Usage:
    python scripts/measure_latency.py --config configs/experiments/<best>.json \\
        --num-samples 25 --output reports/metrics/latency_<name>.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import read_jsonl  # noqa: E402
from medical_rag_thesis.generation import (  # noqa: E402
    build_chat_prompt,
    load_vllm_model,
    make_prompt,
    make_self_feedback_prompt,
    strip_thinking_text,
)
from medical_rag_thesis.prompts import SYSTEM_PROMPTS, SYSTEM_PROMPT_ES, query_text  # noqa: E402
from medical_rag_thesis.retrieval import EmbeddingRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Experiment config JSON (the chosen best).")
    p.add_argument("--num-samples", type=int, default=25,
                   help="Records to time. ~25 is enough for a stable mean.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Untimed warmup requests (exclude cold-start / CUDA-graph capture).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


def generate_one_timed(llm: Any, prompt: str, tokenizer: Any, cfg: dict) -> tuple[str, float, str | None]:
    """One request, timed. A list of length 1 forces vLLM to serve it alone -- no
    batching overlap -- which is what makes this single-request latency rather than
    throughput."""
    from vllm import SamplingParams
    from vllm.sampling_params import RepetitionDetectionParams

    max_new = int(cfg.get("max_new_tokens", 2048))
    max_model_len = int(cfg.get("max_model_len", 16384))
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    allowed = max_model_len - max_new
    if len(ids) > allowed:
        prompt = tokenizer.decode(ids[-allowed:], skip_special_tokens=False)

    sp: dict[str, Any] = dict(
        max_tokens=max_new,
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        top_k=int(cfg.get("top_k", 0)) or -1,
        min_p=float(cfg.get("min_p", 0.0)),
        repetition_penalty=1.0 + float(cfg.get("presence_penalty", 0.0)),
        seed=int(cfg.get("seed", 42)),
    )
    if cfg.get("thinking_token_budget") is not None:
        sp["thinking_token_budget"] = int(cfg["thinking_token_budget"])
    if cfg.get("repetition_detection_max_pattern"):
        sp["repetition_detection"] = RepetitionDetectionParams(
            max_pattern_size=int(cfg["repetition_detection_max_pattern"]),
            min_pattern_size=int(cfg.get("repetition_detection_min_pattern", 1)),
            min_count=int(cfg.get("repetition_detection_min_count", 8)),
        )

    start = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(**sp))  # length-1 list == batch size 1
    elapsed = time.perf_counter() - start
    o = out[0].outputs[0]
    return o.text.strip(), elapsed, o.finish_reason


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    language = cfg.get("language", "es")
    native_thinking = "qwen3" in cfg["model"].lower()
    enable_thinking = bool(cfg.get("think")) if native_thinking else None

    records = read_jsonl(ROOT / cfg["input"])[: args.num_samples + args.warmup]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=bool(cfg.get("trust_remote_code")))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = load_vllm_model(
        cfg["model"],
        dtype=cfg.get("dtype", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code")),
        max_model_len=int(cfg.get("max_model_len", 16384)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.90)),
        reasoning_parser=cfg.get("reasoning_parser"),
    )

    retriever = reranker = None
    if cfg.get("retrieval_index") and int(cfg.get("retrieval_top_k", 0)) > 0:
        # CPU encoder, as in every run in this thesis (shares the GPU with vLLM).
        retriever = EmbeddingRetriever(cfg["retrieval_index"], device=cfg.get("retriever_device", "cpu"))
        if cfg.get("reranker_model") and int(cfg.get("reranker_top_k", 0)) > 0:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(cfg["reranker_model"], device=cfg.get("reranker_device", "cpu"))

    def one_record(record: dict) -> float:
        """Full end-to-end single-request wall time: retrieval + generation (+SF)."""
        t0 = time.perf_counter()
        docs: list[dict] = []
        if retriever is not None:
            q = query_text(record, language=language)
            docs = retriever.query(q, top_k=int(cfg["retrieval_top_k"]), exclude_id=record.get("id"))
            if reranker is not None and docs:
                pairs = [(q, str(d.get("text") or "")) for d in docs]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
                docs = [dict(d) for d, _ in ranked[: int(cfg["reranker_top_k"])]]
        prompt = make_prompt(
            tokenizer, record, documents=docs, no_think=not cfg.get("think"),
            prompt_style=cfg.get("prompt_style", "extractive"), language=language,
            enable_thinking=enable_thinking,
        )
        answer, _, _ = generate_one_timed(llm, prompt, tokenizer, cfg)
        if cfg.get("self_feedback"):
            sf_prompt = make_self_feedback_prompt(
                tokenizer, record,
                answer=strip_thinking_text(answer) if native_thinking else answer,
                documents=docs, prompt_style=cfg.get("prompt_style", "extractive"),
                language=language, enable_thinking=enable_thinking,
            )
            generate_one_timed(llm, sf_prompt, tokenizer, cfg)
        return time.perf_counter() - t0

    # Warmup (cold start, CUDA-graph capture) is excluded from the reported latency.
    for record in records[: args.warmup]:
        one_record(record)

    latencies = []
    for i, record in enumerate(records[args.warmup:], start=1):
        dt = one_record(record)
        latencies.append(dt)
        print(f"  [{i}/{args.num_samples}] {record.get('id','')}: {dt:.2f}s", flush=True)

    latencies.sort()
    summary = {
        "config": Path(args.config).stem,
        "model": cfg["model"],
        "language": language,
        "num_samples": len(latencies),
        "self_feedback": bool(cfg.get("self_feedback")),
        "batch_size": 1,
        "single_request_latency_seconds": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p90": latencies[min(len(latencies) - 1, int(0.90 * len(latencies)))],
            "p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
            "min": latencies[0],
            "max": latencies[-1],
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        },
        "note": ("End-to-end wall time for a SINGLE request (retrieval + generation "
                 "+ self-feedback if enabled), served one at a time. This is the "
                 "user-facing latency; the dev ablation's sec/sample is batched "
                 "throughput and is systematically lower."),
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    s = summary["single_request_latency_seconds"]
    print(f"\n  single-request latency: mean {s['mean']:.2f}s  median {s['median']:.2f}s  "
          f"p90 {s['p90']:.2f}s  p95 {s['p95']:.2f}s")
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
