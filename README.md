# Medical RAG MA thesis

This repository is structured around the first experiments from `plan.pdf`:

1. Normalize and split `SNS1064` into train/dev/test.
2. Normalize a CasiMedicos-style dataset into the same QA schema.
3. Run Spanish `no_rag` / `no_think` LLM baselines.
4. Add random few-shot examples and specialized retrieval context.
5. Evaluate generated answers with lightweight metrics.

## Layout

- `data/raw/`: original datasets, kept out of git.
- `data/interim/`: temporary converted files.
- `data/processed/`: normalized JSONL/CSV splits.
- `src/medical_rag_thesis/`: reusable experiment code.
- `scripts/`: command-line entry points for data prep and experiments.
- `configs/`: starter experiment settings.
- `experiments/runs/`: generated predictions and run artifacts.
- `reports/`: metrics, tables, and figures.
- `docs/`: notes extracted from the thesis plan.

## Supervisor meeting materials

- [Supervisor meeting notes](docs/supervisor_meeting_notes.md)
- [Dev ablation results](reports/metrics/dev_ablation_results.md)
- [Agentic Reasoner dev results](reports/metrics/agentic_dev_results.md)
- [Reading list](docs/reading_list.md)

## Quick start

Use the current environment, or install the package in editable mode:

```bash
python -m pip install -e .
```

Prepare `SNS1064`:

```bash
python scripts/prepare_sns1064.py \
  --input data/raw/sns1064.csv \
  --output-dir data/processed/sns1064 \
  --seed 42
```

Import and normalize the Spanish CasiMedicos argument-mining dataset from
Hugging Face:

```bash
python scripts/import_casimedicos_arg.py \
  --raw-dir data/raw/casimedicos_arg \
  --output-dir data/processed/casimedicos
```

Normalize a local CasiMedicos-style table:

```bash
python scripts/format_casimedicos.py \
  --input data/raw/casimedicos.csv \
  --output data/processed/casimedicos/all.jsonl
```

Create the amplified Spanish dataset (`SNS1064` plus CasiMedicos):

```bash
python scripts/create_amplified_dataset.py \
  --sns-dir data/processed/sns1064 \
  --casimedicos data/processed/casimedicos/all.jsonl \
  --output-dir data/processed/sns1064_casimedicos
```

The amplified dataset preserves the existing SNS1064 splits and assigns
CasiMedicos examples to train/dev/test if they do not already have a `split`
column.

Run the first LLM-only dev experiment:

```bash
python scripts/run_generation_experiment.py \
  --input data/processed/sns1064/dev.jsonl \
  --output experiments/runs/00_mistral7b_no_rag_no_think_dev/predictions.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --experiment-name mistral7b_no_rag_no_think
```

Build a specialized retrieval index from the training split:

```bash
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064/train.jsonl \
  --output-dir models/retrieval/sns1064_train_multilingual_minilm
```

By default, retrieval documents render `topic`, `question`, `subquestion`,
`short_answer`, and `evidence` as labeled text at runtime.

For a no-download retrieval smoke test, add `--backend tfidf`.

Run a retrieval-augmented dev experiment:

```bash
python scripts/run_generation_experiment.py \
  --input data/processed/sns1064/dev.jsonl \
  --output experiments/runs/01_mistral7b_rag_no_think_dev/predictions.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --experiment-name mistral7b_rag_no_think \
  --retrieval-index models/retrieval/sns1064_train_multilingual_minilm \
  --retrieval-top-k 3
```

Evaluate predictions:

```bash
python scripts/evaluate_predictions.py \
  --predictions experiments/runs/00_mistral7b_no_rag_no_think_dev/predictions.jsonl \
  --output reports/metrics/00_mistral7b_no_rag_no_think_dev.json
```

Use `--semantic-model ""` to skip embedding cosine during quick local smoke tests. Add
`--semantic-model intfloat/multilingual-e5-large` to match the cosine-similarity
metric used in the previous SNS RAG prototype.

Quality metrics are written on a 0-100 scale. Add
`--bertscore-model bert-base-multilingual-cased --bertscore-lang es` to compute
section-wise BERTScore F1 over `short_answer` and `evidence`.

For quick smoke tests, add `--limit 5` to generation commands.

Generation and retrieval-index runs write `run.log` and `run.err` in their artifact
directory. Metric evaluations write `<metrics-name>.log` and `<metrics-name>.err`
next to the metrics JSON.

## Slurm runs

On the AZTI server, run GPU experiments through Slurm:

```bash
sbatch slurm/dev_ablation_generation.sh
```

Then submit evaluation after generation succeeds:

```bash
sbatch --dependency=afterok:<generation_job_id> slurm/dev_ablation_evaluation.sh
```

Slurm logs go to `experiments/slurm_logs/`. Experiment logs still go inside each
run directory in `experiments/runs/`.
