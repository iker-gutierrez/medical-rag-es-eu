# Evaluation Metrics

Recommended reporting split:

- Primary answer quality metric: semantic embedding cosine plus token-overlap F1.
- Secondary semantic metric: BERTScore F1, if `bert-score` is installed and the selected model works well for Spanish medical text.
- Retrieval-grounded quality: RAGAS faithfulness, answer relevancy, context precision, and context recall, once an evaluator LLM and embeddings are configured.
- Cheap RAG proxies: answer-context token F1 and gold-context token F1, mainly for debugging retrieved context coverage.
- Efficiency: total run time, mean generation time per example, and input/output token counts.

Quality metrics are reported on a 0-100 scale in the generated metrics JSON files.
BERTScore is computed section-wise over the normalized `short_answer` and `evidence`
columns.

It is reasonable to have two semantic similarity metrics between prediction and gold answer. They are not redundant if they answer slightly different questions:

- Embedding cosine measures sentence-level closeness with one vector per answer. It is cheap, stable, and easy to interpret across experiments.
- BERTScore F1 measures token-level semantic alignment using contextual embeddings. It can be more sensitive to partial matches and wording differences.

For the thesis tables, avoid giving every metric equal status. Use one semantic metric as the main score, keep token-overlap F1 as the lexical anchor, and put BERTScore/RAGAS in secondary columns or appendix tables if they agree with the main findings.
