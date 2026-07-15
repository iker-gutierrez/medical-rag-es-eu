#!/bin/bash
#SBATCH --job-name=seed-check
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/seed_check_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/seed_check_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Two questions that decide whether the seed fix is correct:
#   Q1 does a FIXED seed with n=3 still give 3 INDEPENDENT samples?
#      (if it returns 3 identical copies, MA-RAG's conflict signal is dead)
#   Q2 do DIFFERENT seeds actually change the output?
#      (if not, the whole seed fix is a no-op and +/-std stays fake)
python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=2048,
          gpu_memory_utilization=0.85, enforce_eager=False)
prompt = ["Zerrendatu hiru sintoma ohiko gripearen kasuan. Erantzun euskaraz:"]

print("\n=== Q1: fixed seed, n=3 -> independent samples? ===")
out = llm.generate(prompt, SamplingParams(n=3, seed=42042, temperature=1.0,
                                          top_p=0.95, max_tokens=60))
texts = [c.text.strip() for c in out[0].outputs]
for i, t in enumerate(texts):
    print(f"  cand{i}: {t[:70]!r}")
uniq = len(set(texts))
print(f"  -> {uniq}/3 distinct. {'OK: independent' if uniq > 1 else 'BROKEN: n collapsed to copies'}")

print("\n=== Q2: different seeds -> different output? ===")
per_seed = {}
for s in (42042, 43042, 44042):
    o = llm.generate(prompt, SamplingParams(n=1, seed=s, temperature=1.0,
                                            top_p=0.95, max_tokens=60))
    per_seed[s] = o[0].outputs[0].text.strip()
    print(f"  seed {s}: {per_seed[s][:70]!r}")
u = len(set(per_seed.values()))
print(f"  -> {u}/3 distinct. {'OK: seeds vary' if u > 1 else 'BROKEN: seed has no effect'}")

print("\n=== Q3: same seed twice -> reproducible? ===")
a = llm.generate(prompt, SamplingParams(n=1, seed=42042, temperature=1.0, top_p=0.95, max_tokens=60))[0].outputs[0].text.strip()
b = llm.generate(prompt, SamplingParams(n=1, seed=42042, temperature=1.0, top_p=0.95, max_tokens=60))[0].outputs[0].text.strip()
print(f"  -> {'OK: reproducible' if a == b else 'WARNING: not reproducible'}")
EOF
