from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence


SYSTEM_PROMPT_ES = "Eres un experto clínico."

SYSTEM_PROMPT_EU = "Aditu klinikoa zara."

SYSTEM_PROMPTS = {"es": SYSTEM_PROMPT_ES, "eu": SYSTEM_PROMPT_EU}

PROMPT_STYLES = ("extractive",)


def format_reference_answer(record: Mapping[str, Any], language: str = "es") -> str:
    if language == "eu":
        parts = [
            ("Erantzun laburra", record.get("short_answer", "")),
            ("Ebidentzia", record.get("evidence", "")),
        ]
    else:
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


def format_question(record: Mapping[str, Any], language: str = "es") -> str:
    question = " ".join(
        str(record.get(field, "") or "").strip()
        for field in QUESTION_FIELDS
        if str(record.get(field, "") or "").strip()
    )
    label = "Galdera" if language == "eu" else "Pregunta"
    return f"{label}: {question}"


def query_text(record: Mapping[str, Any], language: str = "es") -> str:
    return format_question(record, language=language)


def format_examples(examples: Sequence[Mapping[str, Any]], language: str = "es") -> str:
    example_label = "Adibidea" if language == "eu" else "Ejemplo"
    options_label = "Aukerak" if language == "eu" else "Opciones"
    blocks = []
    for idx, example in enumerate(examples, start=1):
        options = format_options(example)
        answer = format_reference_answer(example, language=language)
        block = [f"{example_label} {idx}", format_question(example, language=language)]
        if options:
            block.append(f"{options_label}:\n{options}")
        block.append(answer)
        blocks.append("\n".join(block))
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
    no_think: bool = True,  # reserved for future reasoning-mode implementation
    prompt_style: str = "extractive",
    language: str = "es",
) -> str:
    prompt_style = validate_prompt_style(prompt_style)
    if language == "eu":
        return build_extractive_user_prompt_eu(record, examples=examples, documents=documents)
    return build_extractive_user_prompt(record, examples=examples, documents=documents)


def build_extractive_user_prompt(
    record: Mapping[str, Any],
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    context_text = format_context_text(documents or [])
    has_context = bool(context_text)
    options = format_options(record)
    question = format_question(record).removeprefix("Pregunta: ").strip()

    if has_context:
        sections = [
            "Tu tarea es responder usando la información del contexto recuperado, sin inventar datos.",
            "Reglas:\n"
            "- Basa tu respuesta en la información del contexto recuperado. NO añadas información externa. COPIA frases exactas cuando sea posible.\n"
            "- Responde en español.\n"
            "- Responde SIEMPRE en este formato:\n\n"
            "Respuesta corta:\n"
            "(texto extraído)\n\n"
            "Evidencia:\n"
            "(texto extraído)"
        ]
    else:
        sections = [
            "Tu tarea es responder la pregunta clínica de forma justificada.",
            "Reglas:\n"
            "- Usa la información de la pregunta (y las opciones de respuesta, si las hay).\n"
            "- Puedes apoyarte en conocimiento clínico general.\n"
            "- NO inventes datos concretos que no puedas justificar.\n"
            "- Responde en español.\n"
            "- Responde SIEMPRE en este formato:\n\n"
            "Respuesta corta:\n"
            "(respuesta breve)\n\n"
            "Evidencia:\n"
            "(justificación)"
        ]
    if examples:
        sections.append("Usa estos ejemplos solo para aprender el formato de salida:\n\n" + format_examples(examples))
    if has_context:
        sections.append("Contexto recuperado:\n" + context_text)
    sections.append("Pregunta:\n" + question)
    if options:
        sections.append("Opciones:\n" + options)
    return "\n\n".join(sections)


def build_extractive_user_prompt_eu(
    record: Mapping[str, Any],
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    context_text = format_context_text(documents or [])
    has_context = bool(context_text)
    options = format_options(record)
    question = format_question(record, language="eu").removeprefix("Galdera: ").strip()

    if has_context:
        sections = [
            "Zure zeregina errekuperatutako testuinguruko informazioa erabiliz erantzutea da, daturik asmatu gabe.",
            "Arauak:\n"
            "- Oinarritu erantzuna errekuperatutako testuinguruko informazioan. EZ gehitu kanpoko informaziorik. KOPIATU esaldi osoak posible denean.\n"
            "- Erantzun euskaraz.\n"
            "- Erantzun BETI honako formatuan:\n"
            "Erantzun laburra:\n"
            "(erauztitako testua)\n\n"
            "Ebidentzia:\n"
            "(erauztitako testua)"
        ]
    else:
        sections = [
            "Zure zeregina galdera klinikoa modu justifikatuan erantzutea da.",
            "Arauak:\n"
            "- Erabili galderaren informazioa (eta erantzun-aukerak, egonez gero).\n"
            "- Medikuntzako ezagutza orokorraz baliatu zaitezke.\n"
            "- EZ asmatu justifika ezin duzun datu zehatzik.\n"
            "- Erantzun euskaraz.\n"
            "- Erantzun BETI honako formatuan:\n\n"
            "Erantzun laburra:\n"
            "(erantzun laburra)\n\n"
            "Ebidentzia:\n"
            "(ebidentzia edo justifikazioa)"
        ]
    if examples:
        sections.append(
            "Erabili adibide hauek soilik irteera-formatua ikasteko:\n\n"
            + format_examples(examples, language="eu")
        )
    if has_context:
        sections.append("Errekuperatutako testuingurua:\n" + context_text)
    sections.append("Galdera:\n" + question)
    if options:
        sections.append("Aukerak:\n" + options)
    return "\n\n".join(sections)


def build_self_feedback_prompt(
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
    prompt_style: str = "extractive",
    language: str = "es",
) -> str:
    prompt_style = validate_prompt_style(prompt_style)
    if language == "eu":
        return build_extractive_self_feedback_prompt_eu(record, answer=answer, documents=documents)
    return build_extractive_self_feedback_prompt(record, answer=answer, documents=documents)


def build_extractive_self_feedback_prompt(
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    context_text = format_context_text(documents or [])
    has_context = bool(context_text)
    options = format_options(record)
    question = format_question(record).removeprefix("Pregunta: ").strip()
    if has_context:
        checks = (
            "Comprueba:\n"
            "- si está basada en el contexto recuperado.\n"
            "- si hay alucinaciones.\n"
            "- si falta información."
        )
        grounding_rule = "- Usa SOLO información del contexto recuperado."
    else:
        checks = (
            "Comprueba:\n"
            "- si responde a la pregunta.\n"
            "- si hay alucinaciones.\n"
            "- si falta información relevante."
        )
        grounding_rule = "- Usa la pregunta, las opciones de respuesta (si existen), y conocimiento clínico general."
    answer_placeholder = "texto extraído" if has_context else "respuesta breve"
    evidence_placeholder = "texto extraído" if has_context else "justificación"
    sections = [
        "Revisa la siguiente respuesta.",
        checks,
        "Reescribe la respuesta mejorada.",
        "Reglas obligatorias:\n"
        "- Responde SOLO con los dos campos indicados: \"Respuesta corta\" y \"Evidencia\".\n"
        "- NO repitas las instrucciones.\n"
        "- Responde en español.\n"
        f"{grounding_rule}",
        (
            "Responde SIEMPRE en este formato:\n\n"
            "Respuesta corta:\n"
            f"({answer_placeholder})\n\n"
            "Evidencia:\n"
            f"({evidence_placeholder})"
        ),
    ]
    if has_context:
        sections.append("Contexto recuperado:\n" + context_text)
    sections.append("Pregunta:\n" + question)
    if options:
        sections.append("Opciones:\n" + options)
    sections.extend(
        [
            "Respuesta original:\n" + answer,
            "Escribe ahora únicamente la respuesta final con los dos campos indicados.",
        ]
    )
    return "\n\n".join(sections)


def build_extractive_self_feedback_prompt_eu(
    record: Mapping[str, Any],
    *,
    answer: str,
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    context_text = format_context_text(documents or [])
    has_context = bool(context_text)
    options = format_options(record)
    question = format_question(record, language="eu").removeprefix("Galdera: ").strip()
    if has_context:
        checks = (
            "Egiaztatu:\n"
            "- errekuperatutako testuinguruan oinarritua dagoen.\n"
            "- aluzinazioak dauden.\n"
            "- informazioa falta den."
        )
        grounding_rule = "- Erabili SOILIK errekuperatutako testuingurutik datorren informazioa."
    else:
        checks = (
            "Egiaztatu:\n"
            "- galderari erantzuten dion.\n"
            "- aluzinazioak dauden.\n"
            "- informazio garrantzitsua falta den."
        )
        grounding_rule = "- Erabili galdera, erantzun-aukerak (egonez gero), eta medikuntzako ezagutza orokorra."
    answer_placeholder = "erauztitako testua" if has_context else "erantzun laburra"
    evidence_placeholder = "erauztitako testua" if has_context else "ebidentzia edo justifikazioa"
    sections = [
        "Ondorengo erantzuna berrikusi.",
        checks,
        "Erantzun hobetua berridatzi.",
        "Arau derrigorrezkoak:\n"
        "- Erantzun SOILIK bi eremu hauekin: \"Erantzun laburra\" eta \"Ebidentzia\".\n"
        "- EZ errepikatu argibideak.\n"
        "- Erantzun euskaraz.\n"
        f"{grounding_rule}",
        (
            "Erantzun BETI honako formatuan:\n\n"
            "Erantzun laburra:\n"
            f"({answer_placeholder})\n\n"
            "Ebidentzia:\n"
            f"({evidence_placeholder})"
        ),
    ]
    if has_context:
        sections.append("Errekuperatutako testuingurua:\n" + context_text)
    sections.append("Galdera:\n" + question)
    if options:
        sections.append("Aukerak:\n" + options)
    sections.extend(
        [
            "Jatorrizko erantzuna:\n" + answer,
            "Idatzi orain azken erantzuna soilik, adierazitako bi eremuekin.",
        ]
    )
    return "\n\n".join(sections)


def parse_answer_sections(text: str) -> dict[str, str]:
    labels = {
        "short_answer": r"(?:respuesta corta|juicio|erantzun laburra)",
        "evidence": r"(?:evidencia|consideraciones|ebidentzia)",
    }
    stop_labels = (
        r"contexto recuperado",
        r"contexto",
        r"pregunta",
        r"opciones",
        r"respuesta original",
        r"respuesta mejorada",
        r"errekuperatutako testuingurua",
        r"galdera",
        r"aukerak",
        r"jatorrizko erantzuna",
        r"erantzun hobetua",
    )
    parsed: dict[str, str] = {}
    spans = []
    for key, label in labels.items():
        for match in re.finditer(rf"(?im)^\s*{label}\s*:\s*", text):
            spans.append((match.start(), match.end(), key))
    for label in stop_labels:
        for match in re.finditer(rf"(?im)^\s*{label}\s*:\s*", text):
            spans.append((match.start(), match.end(), None))
    spans.sort()
    for idx, (_, value_start, key) in enumerate(spans):
        if key is None:
            continue
        value_end = spans[idx + 1][0] if idx + 1 < len(spans) else len(text)
        value = text[value_start:value_end].strip()
        if value:
            parsed[key] = "\n\n".join(part for part in [parsed.get(key, ""), value] if part)
    if not parsed:
        parsed["short_answer"] = text.strip()
    return parsed
