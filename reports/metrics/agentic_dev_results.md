# Agentic dev results

Dev-only Agentic Reasoner experiment. The judge receives:
- the baseline LLM-only answer
- the answer from the best non-baseline configuration for that dev set
- the question
- the retrieved context from the best non-baseline run, when available

Then the Agentic Reasoner generates a final answer by comparing both candidate answers and prioritizing the better-supported information.

| dev set | agentic BERT | Δ BERT vs best | agentic Cosim | Δ Cosim vs best | agentic sec/sample | Δ sec vs best | agentic tokens/sample | Δ tokens vs best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SNS1064 dev | 75.90 | -0.82 | 88.15 | -0.80 | 11.81 | +5.37 | 5225 | +881 |
| CasiMedicos dev | 79.19 | -0.22 | 91.19 | -0.28 | 11.47 | +4.25 | 4474 | +908 |
| SNS1064+CasiMedicos dev | 76.46 | -1.13 | 88.77 | -0.84 | 12.09 | +4.95 | 6023 | +893 |

## SNS1064 dev

| system | section | Cosim | BERTScore | token F1 | ROUGE-L | sec/sample | tokens/sample |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline LLM only | answer | 78.13 | 61.34 | 2.95 | 2.74 | 9.97 | 1002 |
| Baseline LLM only | evidence | 83.97 | 65.49 | 20.08 | 13.45 | 9.97 | 1002 |
| Baseline LLM only | overall | 81.05 | 63.41 | 11.51 | 8.09 | 9.97 | 1002 |
| Best non-baseline: exp 8 (3-shot + rerank top5) | answer | 89.39 | 79.37 | 38.29 | 38.21 | 6.44 | 4344 |
| Best non-baseline: exp 8 (3-shot + rerank top5) | evidence | 88.52 | 74.06 | 37.62 | 31.56 | 6.44 | 4344 |
| Best non-baseline: exp 8 (3-shot + rerank top5) | overall | 88.95 | 76.72 | 37.95 | 34.88 | 6.44 | 4344 |
| Agentic: baseline vs exp 8 | answer | 88.33 | 79.72 | 34.88 | 34.80 | 11.81 | 5225 |
| Agentic: baseline vs exp 8 | evidence | 87.98 | 72.07 | 35.06 | 28.62 | 11.81 | 5225 |
| Agentic: baseline vs exp 8 | overall | 88.15 | 75.90 | 34.97 | 31.71 | 11.81 | 5225 |

## CasiMedicos dev

| system | section | Cosim | BERTScore | token F1 | ROUGE-L | sec/sample | tokens/sample |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline LLM only | answer | 93.23 | 86.27 | 55.44 | 53.62 | 8.00 | 1304 |
| Baseline LLM only | evidence | 87.73 | 69.49 | 26.27 | 15.55 | 8.00 | 1304 |
| Baseline LLM only | overall | 90.48 | 77.88 | 40.85 | 34.58 | 8.00 | 1304 |
| Best non-baseline: exp 2 (e5 top3) | answer | 94.43 | 88.30 | 64.29 | 62.90 | 7.22 | 3566 |
| Best non-baseline: exp 2 (e5 top3) | evidence | 88.50 | 70.52 | 27.00 | 16.36 | 7.22 | 3566 |
| Best non-baseline: exp 2 (e5 top3) | overall | 91.47 | 79.41 | 45.65 | 39.63 | 7.22 | 3566 |
| Agentic: baseline vs exp 2 | answer | 94.78 | 89.07 | 65.84 | 64.55 | 11.47 | 4474 |
| Agentic: baseline vs exp 2 | evidence | 87.60 | 69.31 | 26.90 | 16.36 | 11.47 | 4474 |
| Agentic: baseline vs exp 2 | overall | 91.19 | 79.19 | 46.37 | 40.45 | 11.47 | 4474 |

## SNS1064+CasiMedicos dev

| system | section | Cosim | BERTScore | token F1 | ROUGE-L | sec/sample | tokens/sample |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline LLM only | answer | 83.29 | 69.86 | 20.88 | 20.12 | 9.27 | 1105 |
| Baseline LLM only | evidence | 85.25 | 66.85 | 22.19 | 14.17 | 9.27 | 1105 |
| Baseline LLM only | overall | 84.27 | 68.36 | 21.54 | 17.14 | 9.27 | 1105 |
| Best non-baseline: exp 8 (3-shot + rerank top5) | answer | 90.87 | 83.12 | 48.30 | 47.71 | 7.14 | 5130 |
| Best non-baseline: exp 8 (3-shot + rerank top5) | evidence | 88.36 | 72.06 | 32.76 | 25.56 | 7.14 | 5130 |
| Best non-baseline: exp 8 (3-shot + rerank top5) | overall | 89.62 | 77.59 | 40.53 | 36.64 | 7.14 | 5130 |
| Agentic: baseline vs exp 8 | answer | 89.98 | 81.95 | 44.75 | 44.13 | 12.09 | 6023 |
| Agentic: baseline vs exp 8 | evidence | 87.56 | 70.97 | 31.10 | 23.18 | 12.09 | 6023 |
| Agentic: baseline vs exp 8 | overall | 88.77 | 76.46 | 37.92 | 33.65 | 12.09 | 6023 |

## Takeaways

- The agentic reasoner did not improve over the selected best non-baseline configuration on overall BERTScore in any dev set.
- The best non-baseline system remains the stronger choice so far: exp 8 for SNS1064, exp 2 for CasiMedicos, and exp 8 for the mixed dev set.
- The agentic reasoner is substantially more expensive because its cost includes generating the baseline answer, generating the best-system answer, and then running the judge/verifier call.
