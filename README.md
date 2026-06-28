# Medical RAG MA thesis

Retrieval-augmented generation (RAG) pipeline for clinical question answering in Spanish and Basque.
The system is evaluated on two tasks: open-answer clinical QA (SNS-1064) and multiple-choice medical exam QA (CasiMédicos-Arg).

## Status (2026-06-29)

- **Spanish dev ablations**: complete. Models: Mistral-7B-Instruct, Qwen3.5-9B (with think/no\_think). Datasets: SNS-1064, CasiMédicos-Arg, Mixed.
- **Basque dev ablations**: complete. Models: Llama-3.1-8B-Instruct, Latxa-Llama-3.1-8B-Instruct. Datasets: SNS-1064 EU, CasiMédicos-Arg EU, Mixed EU.
- **Qwen3.5-4B vs 9B comparison**: complete on Spanish dev.
- **Test set evaluation**: pending supervisor approval.
- **Thesis manuscript**: draft available in Prism (gitignored).

## Repository layout

- `data/raw/`: original datasets, kept out of git.
- `data/interim/`: temporary converted files.
- `data/processed/`: normalized JSONL/CSV splits (Spanish and Basque).
- `src/medical_rag_thesis/`: reusable experiment code (retrieval, generation, evaluation).
- `scripts/`: command-line entry points for data prep and experiments.
- `slurm/`: Slurm job scripts for the AZTI server.
- `experiments/runs/`: generated predictions and run artifacts.
- `reports/metrics/`: ablation result tables and summaries.
- `docs/`: supervisor meeting notes, reading list, bibliography notes.

## Supervisor meeting materials

- [Meeting 2 notes (2026-06-29)](docs/supervisor_meeting2_notes.md)
- [Meeting 1 notes (2026-06-15)](docs/supervisor_meeting_notes.md)
- [Spanish dev ablation results](reports/metrics/es_dev_ablation_results.md)
- [Basque dev ablation results](reports/metrics/eu_dev_ablation_results.md)


## Key results

| Task | Best model | Best config | ROUGE-L |
|------|-----------|-------------|---------|
| Spanish SNS-1064 | Qwen3.5-9B | 3-shot + rerank5, no\_think, noSF | 38.68 |
| Spanish CasiMédicos-Arg | Qwen3.5-9B | 3-shot, no RAG, no\_think, SF | 51.55 |
| Basque SNS-1064 EU | Latxa-Llama-3.1-8B | 3-shot + rerank5, noSF | 35.76 |
| Basque CasiMédicos-Arg EU | Latxa-Llama-3.1-8B | 3-shot + rerank5, noSF | 28.47 |

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
  --model Qwen/Qwen2.5-7B-Instruct \
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

## Slurm runs (AZTI server)

```bash
sbatch slurm/dev_ablation_generation.sh
sbatch --dependency=afterok:<job_id> slurm/dev_ablation_evaluation.sh
```

Slurm logs go to `experiments/slurm_logs/`.
