from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence


SYSTEM_PROMPT_ES = "Eres un experto médico."

SYSTEM_PROMPT_EU = "Aditu medikoa zara."

SYSTEM_PROMPTS = {"es": SYSTEM_PROMPT_ES, "eu": SYSTEM_PROMPT_EU}

# mistralai/Ministral-3-8B-Reasoning-2512's own recommended system prompt
# (its SYSTEM_PROMPT.txt / chat_template.jinja default), verbatim. The model
# card recommends appending it to a custom system prompt rather than
# replacing either one; since this project builds prompts as a flat text
# string rather than through the tokenizer's chat template (see
# generation.build_chat_prompt, which only reaches for apply_chat_template
# when a system message is present -- passing OUR OWN system message means
# the template's own default reasoning-instructions text, which only fires
# when no system role is supplied, never appears at all unless appended
# explicitly here), this has to be added by hand for that model specifically.
MINISTRAL_REASONING_SYSTEM_PROMPT = (
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at a "
    "response. Format your response using Markdown, and use LaTeX for any "
    "mathematical equations. Write both your thoughts and the response in the "
    "same language as the input.\n\n"
    "Your thinking process must follow the template below:[THINK]Your thoughts "
    "or/and draft, like working through an exercise on scratch paper. Be as "
    "casual and as long as you want until you are confident to generate the "
    "response to the user.[/THINK]Here, provide a self-contained response."
)

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


MC_FORMAT_RULE_ES = (
    "- Si la pregunta tiene opciones de respuesta, en el apartado de \"Respuesta corta\" incluye "
    "solo el índice de la opción elegida (ej. \"3.\") y el texto de esa opción. Si la pregunta no "
    "tiene opciones de respuesta, en el apartado de \"Respuesta corta\" responde directamente con "
    "tus palabras."
)
MC_FORMAT_RULE_EU = (
    "- Galderak erantzun-aukerak baditu, \"Erantzun laburra\" atalean sartu soilik aukeratutako "
    "aukeraren indizea (adib. \"3.\") eta aukera horren testua. Galderak erantzun-aukerarik ez badu, "
    "\"Erantzun laburra\" atalean erantzun zuzenean zure hitzekin."
)
PLACEHOLDER_NOTE_ES = (
    "Los símbolos < > delimitan marcadores de posición que especifican el tipo de contenido "
    "esperado. En tu respuesta, sustituye esos marcadores de posición y sus símbolos "
    "delimitadores por el contenido esperado."
)
PLACEHOLDER_NOTE_EU = (
    "< > sinboloek espero den eduki-mota adierazten duten leku-markak mugatzen dituzte. Zure "
    "erantzunean, ordezkatu leku-marka horiek eta haien muga-sinboloak espero den edukiarekin."
)


def short_answer_placeholder(record: Mapping[str, Any], language: str = "es") -> str:
    has_options = bool(record.get("options"))
    if language == "eu":
        return "<aukeraren zenbakia. aukeraren testua>" if has_options else "<sortutako erantzun laburra>"
    return "<número de opción. texto de opción>" if has_options else "<respuesta breve generada>"


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
    short_answer = short_answer_placeholder(record)
    mc_format_rule = f"{MC_FORMAT_RULE_ES}\n" if options else ""

    if has_context:
        sections = [
            "Tu tarea es responder a la pregunta médica de forma justificada.",
            "Reglas:\n"
            "- Basa tu respuesta en la información del contexto extraído. Puedes copiar frases exactas cuando el "
            "contexto extraído esté directamente relacionado con la pregunta. Si el contexto extraído es "
            "insuficiente para una respuesta completa, puedes combinar tu conocimiento médico con el "
            "contexto extraído.\n"
            "- NO inventes datos que no estén respaldados por el contexto extraído o por tu conocimiento "
            "médico.\n"
            f"{mc_format_rule}"
            "- Responde en español."
        ]
    else:
        sections = [
            "Tu tarea es responder a la pregunta médica de forma justificada.",
            "Reglas:\n"
            "- Usa la información de la pregunta (y las opciones de respuesta, si las hay).\n"
            "- Apóyate en tu conocimiento médico.\n"
            "- NO inventes datos que no puedas justificar con tu conocimiento médico.\n"
            f"{mc_format_rule}"
            "- Responde en español."
        ]
    if examples:
        sections.append("Usa estos ejemplos solo para aprender el formato de salida:\n\n" + format_examples(examples))
    if has_context:
        sections.append("Contexto extraído:\n" + context_text)
    sections.append("Pregunta:\n" + question)
    if options:
        sections.append("Opciones:\n" + options)
    evidence_placeholder = (
        "<evidencia basada en el contexto extraído y, si también lo has usado, en tu conocimiento médico>"
        if has_context
        else "<evidencia basada en tu conocimiento médico>"
    )
    sections.append(
        "Responde en este formato, sin incluir texto fuera de los campos indicados:\n\n"
        "Respuesta corta:\n"
        f"{short_answer}\n\n"
        "Evidencia:\n"
        f"{evidence_placeholder}"
    )
    sections.append(PLACEHOLDER_NOTE_ES)
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
    short_answer = short_answer_placeholder(record, language="eu")
    mc_format_rule = f"{MC_FORMAT_RULE_EU}\n" if options else ""

    if has_context:
        sections = [
            "Zure ataza galdera medikoari modu justifikatuan erantzutea da.",
            "Arauak:\n"
            "- Oinarritu erantzuna erauzitako testuinguruko informazioan. Kopiatu ditzakezu esaldi osoak "
            "erauzitako testuingurua galderarekin zuzenki erlazionatuta dagoenean. Erauzitako "
            "testuingurua erantzun oso baterako nahikoa ez bada, zure medikuntza-ezagutza erauzitako "
            "testuinguruarekin konbina dezakezu.\n"
            "- EZ asmatu erauzitako testuinguruan edo zure medikuntza-ezagutzan oinarrituta ez dagoen "
            "daturik.\n"
            f"{mc_format_rule}"
            "- Erantzun euskaraz."
        ]
    else:
        sections = [
            "Zure ataza galdera medikoari modu justifikatuan erantzutea da.",
            "Arauak:\n"
            "- Erabili galderaren informazioa (eta erantzun-aukerak, egonez gero).\n"
            "- Baliatu zure medikuntza-ezagutzaz.\n"
            "- EZ asmatu zure medikuntza-ezagutzarekin justifikatu ezin duzun daturik.\n"
            f"{mc_format_rule}"
            "- Erantzun euskaraz."
        ]
    if examples:
        sections.append(
            "Erabili adibide hauek soilik irteera-formatua ikasteko:\n\n"
            + format_examples(examples, language="eu")
        )
    if has_context:
        sections.append("Erauzitako testuingurua:\n" + context_text)
    sections.append("Galdera:\n" + question)
    if options:
        sections.append("Aukerak:\n" + options)
    evidence_placeholder = (
        "<erauzitako testuinguruan oinarritutako ebidentzia eta, erabili baduzu, zure "
        "medikuntza-ezagutzan oinarritutakoa ere>"
        if has_context
        else "<zure medikuntza-ezagutzan oinarritutako ebidentzia>"
    )
    sections.append(
        "Erantzun honako formatuan, testurik gehitu gabe adierazitako eremuetatik kanpo:\n\n"
        "Erantzun laburra:\n"
        f"{short_answer}\n\n"
        "Ebidentzia:\n"
        f"{evidence_placeholder}"
    )
    sections.append(PLACEHOLDER_NOTE_EU)
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
    short_answer = short_answer_placeholder(record)
    mc_format_rule = f"{MC_FORMAT_RULE_ES}\n" if options else ""
    if has_context:
        checks = (
            "Comprueba:\n"
            "- si está basada en el contexto extraído (y en tu conocimiento médico, si lo has usado).\n"
            "- si hay alucinaciones.\n"
            "- si falta información."
        )
        rules = (
            "Reescribe una respuesta mejorada. Reglas:\n"
            "- Basa tu respuesta en la información del contexto extraído. Puedes copiar frases exactas cuando el "
            "contexto extraído esté directamente relacionado con la pregunta. Si el contexto extraído es "
            "insuficiente para una respuesta completa, puedes combinar tu conocimiento médico con el "
            "contexto extraído.\n"
            "- NO inventes datos que no estén respaldados por el contexto extraído o por tu conocimiento "
            "médico.\n"
            f"{mc_format_rule}"
            "- Responde en español."
        )
        evidence_placeholder = (
            "<evidencia basada en el contexto extraído y, si también lo has usado, en tu conocimiento médico>"
        )
    else:
        checks = (
            "Comprueba:\n"
            "- si está basada en tu conocimiento médico.\n"
            "- si hay alucinaciones.\n"
            "- si falta información relevante."
        )
        rules = (
            "Reescribe una respuesta mejorada. Reglas:\n"
            "- Usa la información de la pregunta (y las opciones de respuesta, si las hay).\n"
            "- Apóyate en tu conocimiento médico.\n"
            "- NO inventes datos que no puedas justificar con tu conocimiento médico.\n"
            f"{mc_format_rule}"
            "- Responde en español."
        )
        evidence_placeholder = "<evidencia basada en tu conocimiento médico>"
    sections = ["Revisa la siguiente respuesta.", checks, rules]
    if has_context:
        sections.append("Contexto extraído:\n" + context_text)
    sections.append("Pregunta:\n" + question)
    if options:
        sections.append("Opciones:\n" + options)
    sections.append("Respuesta inicial:\n" + answer)
    sections.append(
        "Escribe únicamente la respuesta mejorada. Responde en este formato, sin incluir texto fuera "
        "de los campos indicados:\n\n"
        "Respuesta corta:\n"
        f"{short_answer}\n\n"
        "Evidencia:\n"
        f"{evidence_placeholder}"
    )
    sections.append(PLACEHOLDER_NOTE_ES)
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
    short_answer = short_answer_placeholder(record, language="eu")
    mc_format_rule = f"{MC_FORMAT_RULE_EU}\n" if options else ""
    if has_context:
        checks = (
            "Egiaztatu:\n"
            "- erauzitako testuinguruan oinarritua dagoen (eta zure medikuntza-ezagutzan, erabili "
            "baduzu).\n"
            "- aluzinazioak dauden.\n"
            "- informazioa falta den."
        )
        rules = (
            "Berridatzi erantzun hobetu bat. Arauak:\n"
            "- Oinarritu erantzuna erauzitako testuinguruko informazioan. Kopiatu ditzakezu esaldi osoak "
            "erauzitako testuingurua galderarekin zuzenki erlazionatuta dagoenean. Erauzitako "
            "testuingurua erantzun oso baterako nahikoa ez bada, zure medikuntza-ezagutza erauzitako "
            "testuinguruarekin konbina dezakezu.\n"
            "- EZ asmatu erauzitako testuinguruak edo zure medikuntza-ezagutzak babesten ez duen "
            "daturik.\n"
            f"{mc_format_rule}"
            "- Erantzun euskaraz."
        )
        evidence_placeholder = (
            "<erauzitako testuinguruan oinarritutako ebidentzia eta, erabili baduzu, zure "
            "medikuntza-ezagutzan oinarritutakoa ere>"
        )
    else:
        checks = (
            "Egiaztatu:\n"
            "- zure medikuntza-ezagutzan oinarrituta dagoen.\n"
            "- aluzinazioak dauden.\n"
            "- informazio garrantzitsua falta den."
        )
        rules = (
            "Berridatzi erantzun hobetu bat. Arauak:\n"
            "- Erabili galderaren informazioa (eta erantzun-aukerak, egonez gero).\n"
            "- Baliatu zure medikuntza-ezagutzaz.\n"
            "- EZ asmatu zure medikuntza-ezagutzarekin justifikatu ezin duzun daturik.\n"
            f"{mc_format_rule}"
            "- Erantzun euskaraz."
        )
        evidence_placeholder = "<zure medikuntza-ezagutzan oinarritutako ebidentzia>"
    sections = ["Ondorengo erantzuna berrikusi.", checks, rules]
    if has_context:
        sections.append("Erauzitako testuingurua:\n" + context_text)
    sections.append("Galdera:\n" + question)
    if options:
        sections.append("Aukerak:\n" + options)
    sections.append("Hasierako erantzuna:\n" + answer)
    sections.append(
        "Idatzi soilik erantzun hobetua. Erantzun honako formatuan, testurik gehitu gabe "
        "adierazitako eremuetatik kanpo:\n\n"
        "Erantzun laburra:\n"
        f"{short_answer}\n\n"
        "Ebidentzia:\n"
        f"{evidence_placeholder}"
    )
    sections.append(PLACEHOLDER_NOTE_EU)
    return "\n\n".join(sections)


def parse_answer_sections(text: str) -> dict[str, str]:
    labels = {
        "short_answer": r"(?:respuesta corta|juicio|erantzun laburra)",
        "evidence": r"(?:evidencia|consideraciones|ebidentzia)",
    }
    stop_labels = (
        r"contexto extraído",
        r"contexto recuperado",
        r"contexto",
        r"pregunta",
        r"opciones",
        r"respuesta inicial",
        r"respuesta original",
        r"respuesta mejorada",
        r"erauzitako testuingurua",
        r"galdera",
        r"aukerak",
        r"hasierako erantzuna",
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
