#!/bin/bash
#SBATCH --job-name=medrag-translate-eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/translate_eu_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/translate_eu_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/home/igutierrez134/med_rag_thesis/.vendor/transformers426:${PYTHONPATH:-}"

EU_TRANSLATION_MODEL="/home/igutierrez134/.cache/huggingface/models--HiTZ--medical_es-eu/snapshots/38899b3feda911b50b6f7c9a380ba420ff99df65"

echo "Translation job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

echo "=== Translation smoke test ==="
python - <<PYEOF
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM

model_name = "$EU_TRANSLATION_MODEL"
texts = ["Desconocido.", "El paciente tiene daño cerebral"]
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
outputs = model.generate(**tokenizer(texts, return_tensors="pt", padding=True), max_length=128, num_beams=6)
translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print(translations)
if any(len(text) > 200 for text in translations):
    raise SystemExit("Translation smoke test failed: output is unexpectedly long")
if any(fragment in translations[0].lower() for fragment in ["iraunkort", "zenbateko", "behintzat"]):
    raise SystemExit("Translation smoke test failed: output looks degenerate")
PYEOF

mkdir -p data/processed/sns1064_eu
mkdir -p data/processed/casimedicos_eu
mkdir -p data/processed/sns1064_casimedicos_eu

echo "=== Translating SNS1064 ==="
python scripts/translate_to_basque.py \
  --input \
    data/processed/sns1064/train.jsonl \
    data/processed/sns1064/dev.jsonl \
    data/processed/sns1064/test.jsonl \
  --output \
    data/processed/sns1064_eu/train.jsonl \
    data/processed/sns1064_eu/dev.jsonl \
    data/processed/sns1064_eu/test.jsonl \
  --model "$EU_TRANSLATION_MODEL" \
  --batch-size 32

echo "=== Translating CasiMedicos ==="
python scripts/translate_to_basque.py \
  --input \
    data/processed/casimedicos/train.jsonl \
    data/processed/casimedicos/dev.jsonl \
  --output \
    data/processed/casimedicos_eu/train.jsonl \
    data/processed/casimedicos_eu/dev.jsonl \
  --model "$EU_TRANSLATION_MODEL" \
  --batch-size 32

echo "=== Building combined SNS1064+CasiMedicos EU dataset ==="
python - <<'PYEOF'
import json
from pathlib import Path

sns_train  = Path("data/processed/sns1064_eu/train.jsonl").read_text(encoding="utf-8").splitlines()
casi_train = Path("data/processed/casimedicos_eu/train.jsonl").read_text(encoding="utf-8").splitlines()
sns_dev    = Path("data/processed/sns1064_eu/dev.jsonl").read_text(encoding="utf-8").splitlines()
casi_dev   = Path("data/processed/casimedicos_eu/dev.jsonl").read_text(encoding="utf-8").splitlines()
sns_test   = Path("data/processed/sns1064_eu/test.jsonl").read_text(encoding="utf-8").splitlines()

out = Path("data/processed/sns1064_casimedicos_eu")
out.mkdir(parents=True, exist_ok=True)
(out / "train.jsonl").write_text("\n".join(sns_train + casi_train) + "\n", encoding="utf-8")
(out / "dev.jsonl").write_text("\n".join(sns_dev + casi_dev) + "\n", encoding="utf-8")
(out / "test.jsonl").write_text("\n".join(sns_test) + "\n", encoding="utf-8")
print(f"Combined train: {len(sns_train) + len(casi_train)} records")
print(f"Combined dev:   {len(sns_dev) + len(casi_dev)} records")
PYEOF

echo "=== Validating translated datasets ==="
python - <<'PYEOF'
import json
from pathlib import Path

paths = [
    Path("data/processed/sns1064_eu/dev.jsonl"),
    Path("data/processed/casimedicos_eu/dev.jsonl"),
    Path("data/processed/sns1064_casimedicos_eu/dev.jsonl"),
]
bad_fragments = ["iraunkort", "zenbateko", "behintzat", "ezagutzera", "iragarki"]
fields = ["topic", "question", "subquestion", "short_answer", "evidence"]

for path in paths:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad_rows = 0
    for row in rows:
        text = "\n".join(str(row.get(field, "")) for field in fields)
        if any(text.lower().count(fragment) >= 10 for fragment in bad_fragments):
            bad_rows += 1
    print(f"{path}: {bad_rows}/{len(rows)} records with repeated-token artifacts")
    if rows and bad_rows / len(rows) > 0.05:
        raise SystemExit(f"Too many degenerate translations in {path}")
PYEOF

echo "Translation complete at $(date)"
