# Supervisor meeting notes

Date: 2026-06-15

## What is ready to discuss

- Spanish dev ablations are implemented and summarized in `reports/metrics/dev_ablation_results.md`.
- Agentic Reasoner dev experiments are implemented and summarized in `reports/metrics/agentic_dev_results.md`.
- The current best Spanish RAG direction is retrieval-augmented extractive prompting with multilingual-e5 retrieval and cross-encoder reranking.
- Self-feedback was tested as a post-generation step. Its quality deltas are small and mixed, while it adds substantial token and time cost.
- The Agentic Reasoner did not clearly improve over the best non-baseline configuration on dev.

## Basque status

Basque experiments should not yet be interpreted from the old results in `dev_ablation_results.md`.

The first Basque translation run was corrupted by a Transformers compatibility issue with `HiTZ/medical_es-eu`. The problem was reproduced on tiny examples and fixed by running the translation model with an isolated Transformers 4.26.1 stack, matching the model card recommendation.

The corrected Basque Slurm chain has been submitted:

- translation
- EU index rebuild
- Latxa dev generation on SNS1064 EU, CasiMedicos EU, and mixed EU
- EU evaluation
- automatic update of `reports/metrics/dev_ablation_results.md`

Until that chain finishes, Basque numbers should be treated as pending.

## Files to show

- `reports/metrics/dev_ablation_results.md`: Spanish ablation tables and takeaways.
- `reports/metrics/agentic_dev_results.md`: Agentic Reasoner dev results.
- `EXPERIMENTS.md`: experiment roadmap and status.
- `docs/reading_list.md`: bibliography and background reading.
- `src/medical_rag_thesis/prompts.py`: final extractive prompt implementation.
- `src/medical_rag_thesis/retrieval.py`: retrieval/indexing implementation.
- `slurm/submit_eu.sh` and `slurm/translate_eu.sh`: corrected Basque pipeline.
