# Medical RAG MA thesis

Retrieval-augmented generation (RAG) pipeline for clinical question answering in Spanish and Basque.
The system is evaluated on two tasks: open-answer clinical QA (SNS-1064) and multiple-choice medical exam QA (CasiMédicos-Exp), across five generator configurations (Mistral-7B, Qwen3.5-9B in non-thinking and thinking mode, Llama-3.1-8B-Instruct, and its Basque-adapted counterpart Latxa-8B) and an eleven-condition ablation grid varying retrieval depth, cross-encoder reranking, few-shot prompting, self-feedback, and domain restriction, plus four inference-only reasoning pipelines drawn from recent literature.

## Status

- **Spanish and Basque dev ablations**: complete, all eleven conditions, three seeds each, decided by MeanQ (mean of ROUGE-L, BERT-F1, MC-accuracy; see `scripts/meanq.py`).
- **Reasoning-pipeline comparison**: complete for both languages.
- **Test set evaluation**: not yet run, pending supervisor approval.
- **Thesis manuscript**: full draft written; not yet finalized (Basque abstract still pending, some sections flagged for supervisor review). Kept outside this repo (see below).

## Key findings

- Retrieval helps every model in both languages substantially; reranking, few-shot prompting, and self-feedback each help only narrowly and inconsistently, and are actively harmful for the weakest models in each language.
- Of the four reasoning pipelines, only one shows a real gain, and only in Spanish, at several times the inference cost of single-pass retrieval; every pipeline underperforms the single-pass baseline in Basque.
- A persistent 23-25 point MeanQ gap between the best achievable Spanish and Basque configurations survives every technique tested.
- Basque language adaptation (Latxa vs. Llama) improves single-pass generation quality but does not extend to self-feedback or multi-step reasoning.

## Best dev configuration per model

| Language | Model | Best config | MeanQ |
|---|---|---|---|
| Spanish | Mistral-7B-Instruct | rerank top 5 | 49.15±1.21 |
| Spanish | Qwen3.5-9B (no-think) | rerank top 5 | 69.79±0.89 |
| Spanish | Qwen3.5-9B (think) | rerank top 5 | 71.34±0.38 |
| Basque | Llama-3.1-8B-Instruct | retrieve top 5 | 46.88±0.83 |
| Basque | Latxa-Llama-3.1-8B | retrieve top 1 | 47.17±2.87 |

Full per-condition results, including cost (seconds/tokens per sample) and self-feedback deltas, are in the ablation reports linked below.

## Repository layout

- `data/raw/`: original datasets, kept out of git.
- `data/interim/`: temporary converted files.
- `data/processed/`: normalized JSONL/CSV splits (Spanish and Basque).
- `src/medical_rag_thesis/`: reusable experiment code (retrieval, generation, evaluation).
- `scripts/`: command-line entry points for data prep, experiments, and result-table/report generation.
- `slurm/`: Slurm job scripts.
- `experiments/runs/`: generated predictions and run artifacts (gitignored).
- `reports/metrics/`: ablation result tables and summaries.
- `docs/`: supervisor meeting notes, reading list, bibliography notes.

The thesis manuscript itself (LaTeX source and compiled PDF) is kept outside this repository and is not tracked in git.

## Ablation results

- [Spanish dev ablation results](reports/metrics/es_dev_ablation_results.md)
- [Basque dev ablation results](reports/metrics/eu_dev_ablation_results.md)

## Supervisor meeting materials

- [Meeting 2 notes (2026-06-29)](docs/supervisor_meeting2_notes.md)
- [Meeting 1 notes (2026-06-15)](docs/supervisor_meeting_notes.md)

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

Import CasiMédicos-Arg from Hugging Face:

```bash
python scripts/import_casimedicos_arg.py \
  --raw-dir data/raw/casimedicos_arg \
  --output-dir data/processed/casimedicos
```

Create the mixed dataset (SNS-1064 + CasiMédicos-Arg):

```bash
python scripts/create_amplified_dataset.py \
  --sns-dir data/processed/sns1064 \
  --casimedicos data/processed/casimedicos/all.jsonl \
  --output-dir data/processed/sns1064_casimedicos
```

### Retrieval index

```bash
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064/train.jsonl \
  --output-dir models/retrieval/sns1064_train_multilingual_e5
```

### Generation experiment

```bash
python scripts/run_generation_experiment.py \
  --input data/processed/sns1064/dev.jsonl \
  --output experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --model Qwen/Qwen3.5-9B \
  --experiment-name qwen9b_rerank5_3shot_noSF \
  --retrieval-index models/retrieval/sns1064_train_multilingual_e5 \
  --retrieval-top-k 5 \
  --rerank \
  --few-shot 3
```

### Evaluation

```bash
python scripts/evaluate_predictions.py \
  --predictions experiments/runs/qwen9b_rerank5_3shot_noSF/predictions.jsonl \
  --output reports/metrics/qwen9b_rerank5_3shot_noSF.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es
```

## Slurm runs

```bash
sbatch slurm/dev_ablation_generation.sh
sbatch --dependency=afterok:<job_id> slurm/dev_ablation_evaluation.sh
```

Slurm logs go to `experiments/slurm_logs/`.
