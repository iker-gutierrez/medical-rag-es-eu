# Supervisor meeting notes

Date: 2026-06-29

## Goal of this meeting

Review dev ablation results and get green light to run test set experiments with the best model configuration per task.

## Since the last meeting (2026-06-15)

### Experiments completed

- **Basque ablation**: full dev ablation for Basque completed with two models — Llama-3.1-8B-Instruct and Latxa-Llama-3.1-8B-Instruct (Basque-adapted). Three datasets: SNS-1064 EU, CasiMédicos-Arg EU, Mixed EU.
- **Qwen3.5-9B Spanish ablation**: extended the Spanish ablation grid to include Qwen3.5-9B with four configurations per experiment: `no_think × noSF`, `no_think × SF`, `think × noSF`, `think × SF`.
- **Qwen3.5-4B vs. 9B comparison**: ran the same best-configuration analysis for Qwen3.5-4B on Spanish dev to assess the quality-cost tradeoff between model sizes.

### Thesis manuscript

A full draft manuscript is available at `manuscript/main.pdf`. It currently includes:

- Project definition, objectives, and hypothesis (§1)
- System description: retrieval pipeline, generation, self-feedback, evaluation metrics (§2)
- SNS-1064 dataset section with construction pipeline, statistics, and evaluation (§3)
- Evaluation datasets section (§4)
- Dev ablation results section with 6 full tables (§5):
  - Spanish: SNS-1064, CasiMédicos-Arg, Mixed
  - Basque: SNS-1064 EU, CasiMédicos-Arg EU, Mixed EU
- Discussion section skeleton with subsection stubs (§6)
- Conclusions stub (§7)

## Key findings from dev ablations

### Spanish (Mistral-7B-Instruct and Qwen3.5-9B)

- Best Spanish configuration overall: **3-shot + rerank top 5, no\_think, noSF** with Qwen3.5-9B (38.68 ROUGE-L on SNS-1064).
- Self-feedback adds token and latency cost with mixed quality deltas — mostly neutral or slightly negative on SNS-1064, slightly positive on CasiMédicos-Arg.
- Qwen3.5-9B `think` mode consistently underperforms `no_think` on all Spanish datasets, with ROUGE-L dropping dramatically (e.g. 0.54 vs 7.63 on SNS-1064 baseline). Thinking mode appears to hurt extractive-format tasks.
- **4B vs 9B**: on SNS-1064, 4B and 9B reach virtually identical ROUGE-L (38.69 vs 38.68) — 4B preferred for cost. On CasiMédicos-Arg, 9B is substantially better (+6.25 ROUGE-L) and worth the extra latency.

### Basque (Llama-3.1-8B-Instruct and Latxa)

- Latxa-Llama-3.1-8B-Instruct outperforms Llama-3.1-8B-Instruct on most Basque configurations, confirming the benefit of Basque-adapted pretraining.
- Best Basque configuration: **3-shot + rerank top 5, noSF** with Latxa (35.76 ROUGE-L on SNS-1064 EU, 28.47 on CasiMédicos-Arg EU).
- Self-feedback is generally harmful for Basque, reducing ROUGE-L by 1–5 points across most configurations.

## Proposed best configurations for test set

| Task | Model | Configuration |
|------|-------|---------------|
| Spanish SNS-1064 (open-answer) | Qwen3.5-9B | 3-shot + rerank top 5, no\_think, noSF |
| Spanish CasiMédicos-Arg (MC) | Qwen3.5-9B | 3-shot, no RAG, no\_think, SF |
| Spanish Mixed | Qwen3.5-9B | rerank top 5, no\_think, SF |
| Basque SNS-1064 EU (open-answer) | Latxa-Llama-3.1-8B | 3-shot + rerank top 5, noSF |
| Basque CasiMédicos-Arg EU (MC) | Latxa-Llama-3.1-8B | 3-shot + rerank top 5, noSF |
| Basque Mixed EU | Latxa-Llama-3.1-8B | 3-shot + rerank top 5, noSF |

## Files to show

- `manuscript/main.pdf`: full thesis draft with all 6 ablation tables.
- `reports/metrics/es_dev_ablation_results.md`: Spanish ablation summary.
- `reports/metrics/eu_dev_ablation_results.md`: Basque ablation summary.
