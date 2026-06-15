# Experiment checklist

Naming convention:

- `no_rag`: the model answers without retrieved documents.
- `rag`: the model receives retrieved documents as context.
- `no_think`: the prompt asks the model not to show step-by-step reasoning.
- `think`: the prompt allows explicit reasoning when a model supports it.

## Assets and setup

- [x] SNS1064 dataset created with 1064 Spanish samples. Status from `plan.pdf`; raw data is not stored in git.
- [x] Initial RAG model exists. Status from `plan.pdf`; this repo now has scripts to reproduce first variants.
- [x] Thesis repo scaffold created.
- [x] SNS1064 split script created.
- [x] CasiMedicos normalization script created.
- [x] LLM generation script created for no-rag, few-shot, and RAG variants.
- [x] Retrieval index script created with dense and TF-IDF backends.
- [x] Lightweight evaluation script created.
- [x] Smoke tests run on tiny fixtures.
- [x] Real SNS1064 placed in `data/raw/`.
- [x] Real SNS1064 split into train/dev/test.
- [x] CasiMedicos dataset collected from `HiTZ/casimedicos-arg`, Spanish subset.
- [x] CasiMedicos normalized from argument-mining tokens/labels into the shared schema.
- [ ] Optional cross-domain medical corpus selected and normalized.
- [x] Amplified Spanish dataset created at `data/processed/sns1064_casimedicos` from SNS1064 plus normalized CasiMedicos.

## Spanish dev experiments

- Earlier Slurm generation job `5882` and evaluation job `5884` are superseded by schema, prompt, and retrieval-index changes.
- Clean rerun queued on 2026-06-09:
  - generation `00_no_rag_no_think`: job `5964`
  - generation `01_rag_no_think`: job `5965`, dependent on `5964`
  - generation `02_no_rag_think`: job `5966`, dependent on `5965`
  - generation `03_rag_think`: job `5967`, dependent on `5966`
  - evaluation: job `5968`, dependent on all four generation jobs

- [x] Mistral 7B Instruct, `no_rag`, `no_think`, zero-shot baseline.
- [x] Mistral 7B Instruct, `rag`, `no_think`, specialized SNS1064 retrieval, top-3 docs.
- [x] Mistral 7B Instruct, `no_rag`, `think`, zero-shot baseline. Current implementation is prompt-only and should be superseded by explicit reasoning-mode runs.
- [x] Mistral 7B Instruct, `rag`, `think`, specialized SNS1064 retrieval, top-3 docs. Current implementation is prompt-only and should be superseded by explicit reasoning-mode runs.
- [ ] Mistral 7B Instruct, `no_rag`, `no_think`, self-feedback before/after. Queued as job `6067`.
- [ ] Mistral 7B Instruct, `rag`, `no_think`, specialized SNS1064 retrieval top-3, self-feedback before/after. Queued as job `6068`.
- [ ] Mistral 7B Instruct, `no_rag`, `think`, self-feedback before/after. Queued as job `6069`.
- [ ] Mistral 7B Instruct, `rag`, `think`, specialized SNS1064 retrieval top-3, self-feedback before/after. Queued as job `6070`.
- [x] Mistral 7B Instruct, `no_rag`, `no_think`, random in-domain 3-shot with self-feedback. Job `6055`.
- [x] Mistral 7B Instruct, `no_rag`, `no_think`, retrieval-selected in-domain 3-shot for format learning with self-feedback. Job `6056`.
- [x] Mistral 7B Instruct, `rag`, `no_think`, specialized SNS1064 retrieval, top-1 doc, with self-feedback. Job `6051`.
- [x] Mistral 7B Instruct, `rag`, `no_think`, specialized SNS1064 retrieval, top-5 docs, with self-feedback. Job `6053`.
- [x] Mistral 7B Instruct, `rag`, `no_think`, specialized SNS1064 retrieval with notebook-style `multilingual-e5-large` embeddings and self-feedback. E5 index job `6052`; generation job `6054`.
- [ ] Mistral 7B Instruct, `rag`, `no_think`, cross-domain medical retrieval, top-3 docs.
- [ ] Mistral 7B Instruct, `rag`, `no_think`, amplified SNS1064 plus CasiMedicos retrieval, top-3 docs.
- [x] Add optional self-feedback inside each generation run and evaluate before-feedback vs. after-feedback predictions.
- [ ] Implement explicit reasoning modes: `no_think`, `private_reasoning`, and `visible_reasoning`.
- [ ] Qwen 3.5 4B, `no_rag`, `think`.
- [ ] Qwen 3.5 4B, `no_rag`, `no_think`.
- [ ] Qwen 3.5 4B, best retrieval setting from dev, `think`.
- [ ] Qwen 3.5 4B, best retrieval setting from dev, `no_think`.
- [ ] Gemma 3 or 4 comparison, time permitting.

## Metrics to report for every generation experiment

- [x] Wall-clock computation time per example and per run.
- [x] Input/output token counts per example, when a tokenizer is loaded.
- [ ] For self-feedback runs, report initial-generation cost, feedback/refinement cost, total cost, and before/after quality deltas.
- [x] Lexical similarity: token precision, token recall, token-overlap F1.
- [x] Lexical/order-sensitive similarity: ROUGE-L F1.
- [x] Semantic similarity: cosine similarity between normalized sentence embeddings.
- [x] Semantic similarity: section-wise BERTScore F1.
- [ ] RAGAS: faithfulness, answer relevancy, context precision, context recall. Requires installing/configuring `ragas` with an evaluator LLM and embeddings.
- [x] RAG context proxy metrics for debugging: answer-context token F1 and gold-context token F1.

## Retrieval diagnostics

- [ ] Report same-clinical-question hit@1, hit@3, hit@5 for train-only retrieval, using SNS1064 `topic` as the main clinical question.
- [ ] Report same-clinical-question precision@3 and precision@5 for train-only retrieval.
- [ ] Keep gold-context token F1 as retrieved-context gold overlap, not as retrieval accuracy.
- [ ] Optionally annotate 30-50 dev queries manually for retrieval relevance and report precision@k or nDCG@k.

## Agentic reasoner dev experiments

- [ ] Agent 1 answer: LLM-only, `no_rag`.
- [ ] Agent 2 answer: best specialized RAG from Spanish dev experiments.
- [ ] Agent 3 answer: broader medical retrieval variant.
- [ ] Agent 4 judge/verifier/critic combines candidate answers into a final answer.
- [ ] Compare agentic final answer against best single-agent baseline on dev.

## Spanish test experiments

- [ ] Freeze best `no_rag` baseline from dev.
- [ ] Freeze best agentic reasoner from dev.
- [ ] Run Spanish test: LLM-only vs. agentic reasoner.
- [ ] Report short-answer metrics.
- [ ] Report evidence metrics.
- [ ] Report qualitative error analysis.

## Basque experiments

- [ ] Translate evaluation dataset from Spanish to Basque with `HiTZ/medical_es-eu`.
- [ ] Validate translated sample quality manually.
- [ ] Run Latxa inference on Basque set.
- [ ] Run Marmoka inference on Basque set, if available.
- [ ] Compare Spanish vs. Basque performance.
- [ ] Report cost vs. efficiency tables.

## Literature and reporting

- [ ] Survey Medical RAG papers in EMNLP 2025/2026.
- [ ] Survey Medical RAG papers in ACL 2025/2026.
- [ ] Survey Medical RAG papers in EACL 2025/2026.
- [ ] Survey Medical RAG papers in NAACL 2025/2026.
- [ ] Define final research questions.
- [ ] Define final thesis contributions.
- [ ] Create final tables for dev experiments.
- [ ] Create final tables for test experiments.
- [ ] Write methodology section.
- [ ] Write results and discussion section.
