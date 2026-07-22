from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from medical_rag_thesis.data_io import read_jsonl, write_jsonl


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FIELD_LABELS = {
    "topic": "Tema",
    "question": "Pregunta",
    "subquestion": "Subpregunta",
    "short_answer": "Respuesta corta",
    "evidence": "Evidencia",
}

FIELD_LABELS_EU = {
    "topic": "Gaia",
    "question": "Galdera",
    "subquestion": "Azpigaldera",
    "short_answer": "Erantzun laburra",
    "evidence": "Ebidentzia",
}


def field_labels(language: str) -> Mapping[str, str]:
    if language == "eu":
        return FIELD_LABELS_EU
    return FIELD_LABELS


def record_to_document_text(record: Mapping[str, Any], fields: Sequence[str], language: str = "es") -> str:
    parts = []
    labels = field_labels(language)
    for field in fields:
        value = record.get(field)
        if value:
            label = labels.get(field, field)
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def default_dense_prefixes(model_name: str) -> tuple[str, str]:
    normalized = model_name.lower()
    if "e5" in normalized:
        return "passage: ", "query: "
    return "", ""


def build_index(
    records: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    backend: str = "dense",
    text_fields: Sequence[str] = ("topic", "question", "subquestion", "short_answer", "evidence"),
    batch_size: int = 32,
    language: str = "es",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    texts = []
    for idx, record in enumerate(records):
        text = record_to_document_text(record, text_fields, language=language)
        if not text:
            continue
        doc = {
            "doc_id": record.get("id") or f"doc_{idx:05d}",
            "source": record.get("source", ""),
            "topic": record.get("topic", ""),
            "question": record.get("question", ""),
            "subquestion": record.get("subquestion", ""),
            "short_answer": record.get("short_answer", ""),
            "evidence": record.get("evidence", ""),
            "text": text,
        }
        metadata.append(doc)
        texts.append(text)

    if not texts:
        raise ValueError("No non-empty documents found for retrieval indexing")

    if backend == "dense":
        from sentence_transformers import SentenceTransformer

        document_prefix, query_prefix = default_dense_prefixes(model_name)
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            [f"{document_prefix}{text}" for text in texts],
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = normalize_embeddings(np.asarray(embeddings, dtype=np.float32))
        np.savez_compressed(output_dir / "index.npz", embeddings=embeddings)
    elif backend == "tfidf":
        vectorizer = TfidfVectorizer().fit(texts)
        matrix = vectorizer.transform(texts)
        with (output_dir / "tfidf.pkl").open("wb") as handle:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, handle)
    else:
        raise ValueError("backend must be 'dense' or 'tfidf'")

    write_jsonl(metadata, output_dir / "metadata.jsonl")
    config = {
        "backend": backend,
        "model_name": model_name,
        "text_fields": list(text_fields),
        "language": language,
        "num_documents": len(metadata),
        "document_prefix": document_prefix if backend == "dense" else "",
        "query_prefix": query_prefix if backend == "dense" else "",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


class EmbeddingRetriever:
    def __init__(self, index_dir: Union[str, Path], device: Optional[str] = None):
        self.index_dir = Path(index_dir)
        config_path = self.index_dir / "config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.metadata = read_jsonl(self.index_dir / "metadata.jsonl")
        self.backend = self.config.get("backend", "dense")
        if self.backend == "dense":
            from sentence_transformers import SentenceTransformer

            self.embeddings = np.load(self.index_dir / "index.npz")["embeddings"]
            self.model = SentenceTransformer(self.config["model_name"], device=device)
            self.vectorizer = None
            self.matrix = None
        elif self.backend == "tfidf":
            with (self.index_dir / "tfidf.pkl").open("rb") as handle:
                payload = pickle.load(handle)
            self.vectorizer = payload["vectorizer"]
            self.matrix = payload["matrix"]
            self.embeddings = None
            self.model = None
        else:
            raise ValueError(f"Unsupported retrieval backend: {self.backend}")

    def query(
        self, text: str, top_k: int = 3, exclude_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if self.backend == "dense":
            query_prefix = self.config.get("query_prefix", "")
            query_embedding = self.model.encode(
                [f"{query_prefix}{text}"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_embedding = normalize_embeddings(np.asarray(query_embedding, dtype=np.float32))
            scores = np.dot(self.embeddings, query_embedding[0])
        else:
            query_vector = self.vectorizer.transform([text])
            scores = (self.matrix @ query_vector.T).toarray().ravel()
        # Self-retrieval exclusion (sec:retrieval): dev and train instances can
        # derive from the same guideline, so a naive top-k search can return the
        # query's own instance and leak the gold answer. Request one extra
        # candidate and drop whichever one matches exclude_id, so exactly top_k
        # genuinely-other passages are always returned.
        fetch_k = top_k + 1 if exclude_id is not None else top_k
        top_indices = np.argsort(-scores)[:fetch_k]
        results = []
        for rank, index in enumerate(top_indices, start=1):
            item = dict(self.metadata[int(index)])
            if exclude_id is not None and str(item.get("doc_id")) == str(exclude_id):
                continue
            item["score"] = float(scores[int(index)])
            item["rank"] = len(results) + 1
            results.append(item)
            if len(results) == top_k:
                break
        return results


# Evidence-only experiment: used only with indices built by indexing solely the
# evidence column of the corpus (not the full record), a rejected alternative
# design discussed but not adopted in the manuscript (see manuscript/main.tex,
# "Corpus and index").
class LeakLoggingEmbeddingRetriever(EmbeddingRetriever):
    """EmbeddingRetriever variant used ONLY by the evidence-only-index runs
    (scripts/clone_configs_evidence_only.py's `log_retrieval_leak` config
    field), to answer a concrete question raised during that work: how often
    would the query's own gold document have appeared in a naive top-k+1
    search, absent self-retrieval exclusion?

    Behaviourally IDENTICAL to EmbeddingRetriever.query() -- same top_k
    passages returned to the generator, same exclusion of the query's own
    document -- this only ADDS bookkeeping: every call over-fetches k+1
    candidates (as the base class already does when exclude_id is set) and
    records, in self.leak_log, whether the excluded document was actually
    present among them. Kept as a separate subclass rather than folded into
    EmbeddingRetriever itself so the shared class used by the live ablation
    grid and reasoning pipelines is untouched.
    """

    def __init__(self, index_dir: Union[str, Path], device: Optional[str] = None):
        super().__init__(index_dir, device=device)
        # One entry per query() call that passed exclude_id: {"query_id":
        # exclude_id, "top_k": top_k, "excluded_present": bool,
        # "excluded_rank": int | None (1-indexed position in the naive top
        # k+1 ranking, before exclusion, only set when excluded_present)}.
        self.leak_log: list[dict[str, Any]] = []

    def query(
        self, text: str, top_k: int = 3, exclude_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if self.backend == "dense":
            query_prefix = self.config.get("query_prefix", "")
            query_embedding = self.model.encode(
                [f"{query_prefix}{text}"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_embedding = normalize_embeddings(np.asarray(query_embedding, dtype=np.float32))
            scores = np.dot(self.embeddings, query_embedding[0])
        else:
            query_vector = self.vectorizer.transform([text])
            scores = (self.matrix @ query_vector.T).toarray().ravel()

        fetch_k = top_k + 1 if exclude_id is not None else top_k
        top_indices = np.argsort(-scores)[:fetch_k]

        if exclude_id is not None:
            excluded_present = False
            excluded_rank = None
            for naive_rank, index in enumerate(top_indices, start=1):
                if str(self.metadata[int(index)].get("doc_id")) == str(exclude_id):
                    excluded_present = True
                    excluded_rank = naive_rank
                    break
            self.leak_log.append({
                "query_id": exclude_id,
                "top_k": top_k,
                "excluded_present": excluded_present,
                "excluded_rank": excluded_rank,
            })

        results = []
        for index in top_indices:
            item = dict(self.metadata[int(index)])
            if exclude_id is not None and str(item.get("doc_id")) == str(exclude_id):
                continue
            item["score"] = float(scores[int(index)])
            item["rank"] = len(results) + 1
            results.append(item)
            if len(results) == top_k:
                break
        return results


def load_records_for_index(paths: Iterable[Union[str, Path]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records
