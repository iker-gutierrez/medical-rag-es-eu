from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence


SYSTEM_PROMPT_ES = (
    "Eres un asistente clínico. Responde en español con una respuesta breve, "
    "y evidencia concreta. No inventes datos que no estén en la pregunta o en el contexto."
)


PROMPT_STYLES = ("generative", "extractive", "v1_extractive")


def format_reference_answer(record: Mapping[str, Any]) -> str:
    parts = [
        ("Respuesta corta", record.get("short_answer", "")),
        ("Evidencia", record.get("evidence", "")),
    ]
    return "\n".join(f"{label}: {value}".strip() for label, value in parts if value)


def format_options(record: Mapping[str, Any]) -> str:
    options = record.get("options")
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
    if isinstance(options, list):
        return "\n".join(str(option) for option in options)
    return str(options)


QUESTION_FIELDS = ("topic", "question", "subquestion")


def format_question(record: Mapping[str, Any]) -> str:
    question = " ".join(
        str(record.get(field, "") or "").strip()
        for field in QUESTION_FIELDS
        if str(record.get(field, "") or "").strip()
    )
    return f"Pregunta: {question}"


def query_text(record: Mapping[str, Any]) -> str:
    return format_question(record)


def format_examples(examples: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for idx, example in enumerate(examples, start=1):
        options = format_options(example)
        answer = format_reference_answer(example)
        block = [f"Ejemplo {idx}", format_question(example)]
        if options:
            block.append(f"Opciones:\n{options}")
        block.append(answer)
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def format_documents(documents: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for idx, document in enumerate(documents, start=1):
        text = document.get("text") or document.get("evidence") or document.get("short_answer") or ""
        topic = document.get("topic")
        prefix = f"Documento {idx}"
        if topic:
            prefix += f" ({topic})"
        blocks.append(f"{prefix}:\n{text}")
    return "\n\n".join(blocks)


def format_context_text(documents: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(
        str(document.get("text") or document.get("evidence") or document.get("short_answer") or "").strip()
        for document in documents
        if str(document.get("text") or document.get("evidence") or document.get("short_answer") or "").strip()
    )


def validate_prompt_style(prompt_style: str) -> str:
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    return prompt_style


def build_user_prompt(
    record: Mapping[str, Any],
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    no_think: bool = True,
    prompt_style: str = "generative",
) -> str:
    prompt_style = validate_prompt_style(prompt_style)
    if prompt_style == "extractive":
        return build_extractive_user_prompt(
            record,
            examples=examples,
            documents=documents,
            no_think=no_think,
        )
    if prompt_style == "v1_extractive":
        return build_v1_extractive_user_prompt(
            record,
            examples=examples,
            documents=documents,
            no_think=no_think,
        )

    sections = []
    if no_think:
        sections.append("No muestres razonamiento paso a paso.")
    if examples:
        sections.append("Usa estos ejemplos solo para aprender el formato de salida:\n\n" + format_examples(examples))
    if documents:
        sections.append("Contexto recuperado:\n\n" + format_documents(documents))

    options = format_options(record)
    task = [format_question(record)]
    if options:
        task.append(f"Opciones:\n{options}")
    task.append(
        "Devuelve exactamente estos campos:\n"
        "Respuesta corta: <respuesta>\n"
        "Evidencia: <evidencia breve>"
    )
    sections.append("\n".join(task))
    return "\n\n".join(sections)


def build_extractive_user_prompt(
    record: Mapping[str, Any],
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    no_think: bool = True,
) -> str:
    context_text = format_context_text(documents or [])
    options = format_options(record)
    question = format_question(record).removeprefix("Pregunta: ").strip()

    sections = [
        "Eres un asistente clínico.",
        "Tu tarea es EXTRAER información del contexto, no inventar ni reformular.",
        (
            "Reglas:\n"
            "- Usa SOLO información del contexto\n"
            "- NO añadas información externa\n"
            "- COPIA frases exactas cuando sea posible\n"
            "- Mantén el idioma en español\n"
            "- Responde SIEMPRE en este formato:\n\n"
            "Respuesta corta:\n"
            "(texto extraído)\n\n"
            "Evidencia:\n"
            "(texto extraído)"
        ),
    ]
    if no_think:
        sections.append("No muestres razonamiento paso a paso.")
    if examples:
        sections.append("Usa estos ejemplos solo para aprender el formato de salida:\n\n" + format_examples(examples))
    sections.append("Contexto:\n" + (context_text if context_text else "Ninguno."))
    task = ["Pregunta:\n" + question]
    if options:
        task.append("Opciones:\n" + options)
    sections.append("\n\n".join(task))
    return "\n\n".join(sections)


def build_v1_extractive_user_prompt(
    record: Mapping[str, Any],
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    no_think: bool = True,
) -> str:
    context_text = format_context_text(documents or [])
    options = format_options(record)
    question = format_question(record).removeprefix("Pregunta: ").strip()
    sections = [
        "Eres un asistente clínico.",
        "Tu tarea es EXTRAER información del contexto, no inventar ni reformular.",
        (
            "Reglas:\n"
            "- Usa SOLO información del contexto\n"
            "- NO añadas información externa\n"
            "- COPIA frases exactas cuando sea posible\n"
            "- Mantén el idioma en español\n"
            "- Responde SIEMPRE en este formato:\n\n"
            "Respuesta corta:\n"
            "(texto extraído)\n\n"
            "Evidencia:\n"
            "(texto extraído)"
        ),
    ]
    if no_think:
        sections.append("No muestres razonamiento paso a paso.")
    if examples:
        sections.append("Usa estos ejemplos solo para aprender el formato de salida:\n\n" + format_examples(examples))
    sections.extend(
        [
            "Contexto:\n" + (context_text if context_text else "Ninguno."),
            "Pregunta:\n" + question,
        ]
    )
    if options:
        sections.append("Opciones:\n" + options)
    return "\n\n".join(sections)


def build_self_feedback_prompt(
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    prompt_style: str = "generative",
) -> str:
    prompt_style = validate_prompt_style(prompt_style)
    if prompt_style == "extractive":
        return build_extractive_self_feedback_prompt(record, answer=answer, documents=documents)
    if prompt_style == "v1_extractive":
        return build_v1_extractive_self_feedback_prompt(record, answer=answer, documents=documents)

    context = format_documents(documents or [])
    options = format_options(record)
    task = [format_question(record)]
    if options:
        task.append(f"Opciones:\n{options}")
    task.append(
        "Revisa la respuesta original.\n"
        "Comprueba si esta basada en el contexto, si hay alucinaciones y si falta informacion.\n"
        "Reescribe una respuesta mejorada en espanol.\n"
        "Devuelve exactamente estos campos:\n"
        "Respuesta corta: <respuesta>\n"
        "Evidencia: <evidencia breve>"
    )
    sections = [
        "Contexto recuperado:\n\n" + context if context else "Contexto recuperado: Ninguno.",
        "\n".join(task),
        "Respuesta original:\n" + answer,
        "Respuesta mejorada:",
    ]
    return "\n\n".join(sections)


def build_extractive_self_feedback_prompt(
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    context_text = format_context_text(documents or [])
    options = format_options(record)
    question = format_question(record).removeprefix("Pregunta: ").strip()
    task = ["Pregunta:\n" + question]
    if options:
        task.append("Opciones:\n" + options)
    sections = [
        "Revisa la siguiente respuesta.",
        (
            "Comprueba:\n"
            "- si está basada en el contexto\n"
            "- si hay alucinaciones\n"
            "- si falta información"
        ),
        "Reescribe la respuesta mejorada.",
        "Responde en español.",
        "Contexto:\n" + (context_text if context_text else "Ninguno."),
        "\n\n".join(task),
        "Respuesta original:\n" + answer,
        "Respuesta mejorada:",
    ]
    return "\n\n".join(sections)


def build_v1_extractive_self_feedback_prompt(
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    context_text = format_context_text(documents or [])
    options = format_options(record)
    question = format_question(record).removeprefix("Pregunta: ").strip()
    sections = [
        "Revisa la siguiente respuesta.",
        (
            "Comprueba:\n"
            "- si está basada en el contexto\n"
            "- si hay alucinaciones\n"
            "- si falta información"
        ),
        "Reescribe la respuesta mejorada.",
        "Responde en español.",
        (
            "Responde SIEMPRE en este formato:\n\n"
            "Respuesta corta:\n"
            "(texto extraído)\n\n"
            "Evidencia:\n"
            "(texto extraído)"
        ),
        "Contexto:\n" + (context_text if context_text else "Ninguno."),
        "Pregunta:\n" + question,
    ]
    if options:
        sections.append("Opciones:\n" + options)
    sections.extend(
        [
            "Respuesta original:\n" + answer,
            "Respuesta mejorada:",
        ]
    )
    return "\n\n".join(sections)


def parse_answer_sections(text: str) -> dict[str, str]:
    labels = {
        "short_answer": r"(?:respuesta corta|juicio)",
        "evidence": r"(?:evidencia|consideraciones)",
    }
    parsed: dict[str, str] = {}
    spans = []
    for key, label in labels.items():
        for match in re.finditer(rf"(?im)^\s*{label}\s*:\s*", text):
            spans.append((match.start(), match.end(), key))
    spans.sort()
    for idx, (_, value_start, key) in enumerate(spans):
        value_end = spans[idx + 1][0] if idx + 1 < len(spans) else len(text)
        value = text[value_start:value_end].strip()
        if value:
            parsed[key] = "\n\n".join(part for part in [parsed.get(key, ""), value] if part)
    if not parsed:
        parsed["short_answer"] = text.strip()
    return parsed
