from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from medical_rag_thesis.prompts import SYSTEM_PROMPTS, SYSTEM_PROMPT_ES, build_self_feedback_prompt, build_user_prompt


def resolve_dtype(dtype: str) -> Any:
    import torch

    if dtype == "auto":
        return "auto"
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_generation_model(
    model_name: str,
    *,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    language: str = "es",
    *,
    enable_thinking: Optional[bool] = None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    answer_label = "Erantzuna" if language == "eu" else "Respuesta"
    return f"{system_prompt}\n\n{user_prompt}\n\n{answer_label}:"


def strip_thinking_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def decode_generated_response(tokenizer: Any, generated: Any, enable_thinking: Optional[bool]) -> str:
    token_ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
    if enable_thinking:
        end_think_ids = []
        for token in ("</think>", "<|im_end_think|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
                end_think_ids.append(token_id)
        end_think_ids.append(151668)
        end_positions = [idx for idx, token_id in enumerate(token_ids) if token_id in set(end_think_ids)]
        if end_positions:
            token_ids = token_ids[end_positions[-1] + 1 :]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return strip_thinking_text(text)


def generate_one(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    enable_thinking: Optional[bool] = None,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Left-truncate if prompt exceeds context window; keeps the question at the end
    max_ctx = getattr(model.config, "max_position_embeddings", 4096)
    allowed_input = max_ctx - max_new_tokens
    if inputs["input_ids"].shape[1] > allowed_input:
        inputs = {k: v[:, -allowed_input:] for k, v in inputs.items()}
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return decode_generated_response(tokenizer, generated, enable_thinking)


def make_prompt(
    tokenizer: Any,
    record: Mapping[str, Any],
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    no_think: bool = True,
    prompt_style: str = "extractive",
    language: str = "es",
    system_prompt: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    resolved_prompt = system_prompt or SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ES)
    user_prompt = build_user_prompt(
        record,
        examples=examples,
        documents=documents,
        no_think=no_think,
        prompt_style=prompt_style,
        language=language,
    )
    return build_chat_prompt(
        tokenizer,
        resolved_prompt,
        user_prompt,
        language=language,
        enable_thinking=enable_thinking,
    )


def make_self_feedback_prompt(
    tokenizer: Any,
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    prompt_style: str = "extractive",
    language: str = "es",
    system_prompt: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    resolved_prompt = system_prompt or SYSTEM_PROMPTS.get(language, SYSTEM_PROMPT_ES)
    user_prompt = build_self_feedback_prompt(
        record,
        answer=answer,
        documents=documents,
        prompt_style=prompt_style,
        language=language,
    )
    return build_chat_prompt(
        tokenizer,
        resolved_prompt,
        user_prompt,
        language=language,
        enable_thinking=enable_thinking,
    )
