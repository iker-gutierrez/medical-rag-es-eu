"""Causal-aware retrieval scoring, following MedCoT-RAG (Wang, Khatibi & Rahmani,
2025, arXiv:2508.15849). The paper defines the retrieval score for a document d
given query q as

    s(d, q) = alpha * sim(q, d) + beta * psi(d)

where sim(q, d) is embedding cosine similarity (MedCPT in the original paper; our
own multilingual-e5-large dense index here, see sec:retrieval) and psi(d) is a
"causal relevance score that estimates the diagnostic utility of the document",
computed by "detecting medically relevant causal patterns in the text, such as
causal operators ('leads to', 'causes', 'mediates'), treatment-action-effect
relations, and mechanistic disease explanations", implemented "via a weighted
keyword matching scheme, normalized by document length to avoid verbosity bias."

The paper gives no numeric value for alpha/beta and no keyword list (its corpus
is English-language PubMed/StatPearls/textbooks/Wikipedia). This module is our
adaptation to the thesis's Spanish and Basque clinical corpora: the three
keyword categories are translated directly from the paper's own examples and
description, alpha = beta = 1.0 (an unweighted sum, since the paper specifies
no other value), and psi(d) is length-normalized exactly as described.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# Default weights for s(d, q) = ALPHA * sim(q, d) + BETA * psi(d). The paper
# never gives concrete values, so we use an unweighted sum of the two terms.
ALPHA = 1.0
BETA = 1.0

# Three categories from the paper's psi(d) description, translated into
# Spanish and Basque for this thesis's corpora:
#   1. causal operators ("leads to", "causes", "mediates")
#   2. treatment-action-effect relations
#   3. mechanistic disease explanations
CAUSAL_KEYWORDS_ES: tuple[str, ...] = (
    # causal operators
    "provoca", "provocan", "causa", "causan", "conduce a", "conducen a",
    "lleva a", "llevan a", "media", "mediado por", "mediada por",
    "desencadena", "desencadenan", "resulta en", "da lugar a", "genera",
    "produce", "producen", "debido a", "como consecuencia de", "por lo que",
    # treatment-action-effect relations
    "reduce el riesgo", "reduce la", "disminuye el riesgo", "disminuye la",
    "aumenta el riesgo", "aumenta la", "mejora la", "empeora la",
    "el tratamiento con", "al administrar", "previene", "evita",
    # mechanistic disease explanations
    "mecanismo", "fisiopatología", "fisiopatológico", "vía metabólica",
    "inhibición de", "inhibe", "activa", "activación de", "receptor",
    "mediante la", "a través de",
)

CAUSAL_KEYWORDS_EU: tuple[str, ...] = (
    # causal operators
    "eragiten du", "eragiten dute", "sortzen du", "sortzen dute",
    "-(e)k eragindako", "ondorioz", "horren ondorioz", "bitartekaritza",
    "bitartekari", "abiarazten du", "eragiten duen",
    # treatment-action-effect relations
    "arriskua murrizten du", "arriskua handitzen du", "hobetzen du",
    "okerragotzen du", "tratamenduak", "administratzean", "prebenitzen du",
    "saihesten du",
    # mechanistic disease explanations
    "mekanismoa", "fisiopatologia", "fisiopatologikoa", "bide metabolikoa",
    "inhibizioa", "inhibitzen du", "aktibatzen du", "aktibazioa",
    "hartzailea", "bidez",
)


def causal_keywords(language: str) -> tuple[str, ...]:
    return CAUSAL_KEYWORDS_EU if language == "eu" else CAUSAL_KEYWORDS_ES


def psi(text: str, language: str = "es") -> float:
    """Causal relevance score: count of causal-pattern keyword matches,
    normalized by document length (in words) to avoid rewarding verbosity."""
    if not text:
        return 0.0
    lowered = text.lower()
    word_count = max(len(lowered.split()), 1)
    hits = sum(len(re.findall(re.escape(kw), lowered)) for kw in causal_keywords(language))
    return hits / word_count


def causal_score(
    query: str,
    documents: Sequence[Mapping[str, Any]],
    *,
    language: str = "es",
    alpha: float = ALPHA,
    beta: float = BETA,
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-scores a candidate pool of already dense-retrieved documents with
    MedCoT-RAG's composite s(d, q) = alpha*sim(q,d) + beta*psi(d), and returns
    the top_k by that composite score. `sim(q, d)` is each document's existing
    dense-retrieval `score` field (already a cosine similarity in [-1, 1] from
    EmbeddingRetriever, sec:retrieval); this function only adds the psi(d) term
    and re-ranks -- it does not re-embed or re-query the index."""
    scored = []
    for doc in documents:
        sim = float(doc.get("score", 0.0))
        causal = psi(str(doc.get("text") or ""), language=language)
        item = dict(doc)
        item["sim_score"] = sim
        item["psi_score"] = causal
        item["causal_composite_score"] = alpha * sim + beta * causal
        scored.append(item)
    scored.sort(key=lambda item: item["causal_composite_score"], reverse=True)
    results = []
    for rank, item in enumerate(scored[:top_k], start=1):
        item = dict(item)
        item["pre_causal_rank"] = item.get("rank")
        item["rank"] = rank
        results.append(item)
    return results
