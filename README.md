# Medical RAG MA thesis

Retrieval-augmented generation (RAG) pipeline for clinical question answering in Spanish and Basque.
The system is evaluated on two tasks: open-answer clinical QA (SNS-1064) and multiple-choice medical exam QA (CasiMédicos-Exp), across four core generator configurations (Qwen3.5-9B in non-thinking and thinking mode, Llama-3.1-8B-Instruct, and its Basque-adapted counterpart Latxa-Llama-3.1-8B-Instruct) and an eleven-condition ablation grid varying retrieval depth, cross-encoder reranking, few-shot prompting, self-feedback, and domain restriction, plus four inference-only reasoning pipelines drawn from recent literature. Ministral-8B is additionally evaluated as a fifth, exploratory single-pass and reasoning-pipeline configuration.

Full system implementation: retrieval, generation, self-feedback, reasoning pipelines, and evaluation.

## Status

- **Spanish and Basque dev ablations**: complete, all eleven conditions, three seeds each, decided by MeanQ (mean of ROUGE-L, BERT-F1, MC-accuracy; see `scripts/meanq.py`).
- **Reasoning-pipeline comparison**: complete for both languages, including a second Basque backbone (Llama) alongside Latxa, and both Qwen3.5-9B modes.
- **Test set evaluation**: not yet run, pending supervisor approval.

## Key findings

- Retrieval helps every model in both languages substantially; reranking and few-shot prompting each help only narrowly and inconsistently, and are actively harmful for the weakest models in each language, while self-feedback is neutral for three of the four models but a genuine, if modest, gain for the Basque-adapted model specifically.
- Of the four reasoning pipelines, two show a real gain over single-pass retrieval, and only for one of the two Spanish configurations tested, at several times the inference cost; every pipeline underperforms the single-pass baseline in Basque, and for the other Spanish configuration.
- A persistent 20-26 point MeanQ gap between the best achievable Spanish and Basque configurations survives every technique tested.
- Basque language adaptation (Latxa vs. Llama) does not raise overall single-pass MeanQ above the non-adapted model's, but it does raise multiple-choice accuracy specifically, and it is the only technique tested for which the Basque-adapted model shows a genuine advantage: a positive self-feedback gain that does not extend to multi-step reasoning.

## Best dev configuration per model

| Language | Model | Best config | MeanQ |
|---|---|---|---|
| Spanish | Qwen3.5-9B (no-think) | rerank top 5 | 69.15±0.72 |
| Spanish | Qwen3.5-9B (think) | 3-shot + rerank top 5 | 73.51±0.25 |
| Basque | Llama-3.1-8B-Instruct | retrieve top 3 | 49.01±1.71 |
| Basque | Latxa-Llama-3.1-8B-Instruct | retrieve top 3 (with self-feedback) | 47.66±0.37 |

Full per-condition results, including cost (seconds/tokens per sample) and self-feedback deltas, are in the ablation reports linked below.

## Repository layout

- `data/raw/`: original datasets, kept out of git.
- `data/interim/`: temporary converted files.
- `data/processed/`: normalized JSONL/CSV splits (Spanish and Basque).
- `src/medical_rag_thesis/`: reusable experiment code (retrieval, generation, evaluation, reasoning pipelines).
- `scripts/`: command-line entry points for data prep, experiments, staged ablation, and result-table/report generation.
- `slurm/`: Slurm job scripts, including the staged-ablation launchers (`slurm/staged_*.sh`).
- `configs/experiments/`: per-run experiment configs (retrieval depth, reranking, few-shot, self-feedback, reasoning pipeline).
- `experiments/runs/`: generated predictions and run artifacts (gitignored).
- `reports/metrics/`: ablation result tables and summaries.
- `docs/`: current prompt reference (`prompts.md`), supervisor meeting notes, reading list, bibliography notes.

The thesis manuscript itself (LaTeX source and compiled PDF) is kept outside this repository and is not tracked in git.

## Ablation results

- [Spanish dev ablation results](reports/metrics/es_dev_ablation_results.md)
- [Basque dev ablation results](reports/metrics/eu_dev_ablation_results.md)

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

### Data preparation

Prepare SNS-1064:

```bash
python scripts/prepare_sns1064.py \
  --input data/raw/sns1064.csv \
  --output-dir data/processed/sns1064 \
  --seed 42
```

Import CasiMédicos-Exp from Hugging Face (`HiTZ/casimedicos-exp`):

```bash
python scripts/import_casimedicos_exp.py \
  --raw-dir data/raw/casimedicos_exp \
  --output-dir data/processed/casimedicos
```

Create the mixed dataset (SNS-1064 + CasiMédicos-Exp):

```bash
python scripts/create_amplified_dataset.py \
  --sns-dir data/processed/sns1064 \
  --casimedicos data/processed/casimedicos/all.jsonl \
  --output-dir data/processed/sns1064_casimedicos
```

### Retrieval index

The retrieval corpus is the full corpus (train, dev and test together), with each query's own gold instance excluded at query time rather than by corpus-splitting:

```bash
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064/all.jsonl \
  --output-dir models/retrieval/sns1064_full_multilingual_e5_large
```

### Generation experiment

```bash
python scripts/run_generation_experiment.py \
  --input data/processed/sns1064/dev.jsonl \
  --output experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --model Qwen/Qwen3.5-9B \
  --experiment-name qwen9b_rerank5_3shot_noSF \
  --retrieval-index models/retrieval/sns1064_full_multilingual_e5_large \
  --retrieval-top-k 15 \
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --reranker-top-k 5 \
  --few-shot-file data/processed/sns1064/train.jsonl \
  --few-shot-k 3
```

In practice, most experiments are launched from a JSON config in `configs/experiments/` via `scripts/run_generation_from_config.py`, which translates config fields into the CLI flags above; see any file under `configs/experiments/` for the full field list.

### Evaluation

```bash
python scripts/evaluate_predictions.py \
  --predictions experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --output reports/metrics/qwen9b_rerank5_3shot_noSF.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es
```

### Reasoning pipelines

```bash
python scripts/run_reasoning_pipeline.py \
  --config configs/experiments/1530_qwen35_9b_no_think_structured_cot_meanq_best_extractive_mixed_dev.json
```

## Slurm runs

The staged ablation grid (retrieval depth -> reranking -> few-shot -> domain restriction) is launched per model/language via the `slurm/staged_*.sh` scripts, e.g.:

```bash
sbatch slurm/staged_qwen35_9b_no_think_A.sh
sbatch slurm/staged_latxa_A.sh
```

Retrieval indices are rebuilt with:

```bash
sbatch slurm/rebuild_full_corpus_indices.sh
```

Slurm logs go to `experiments/slurm_logs/`.
