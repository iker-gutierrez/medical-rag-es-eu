# reading list

This is the working literature list for the thesis. Read the priority A papers first, then use priority B and C to fill the related work, methods, and discussion sections.

The list has been updated to include recent 2025/26 work on medical RAG, RAG evaluation, medical QA benchmarks, Spanish medical LLMs, and current open LLMs. Some older or more generic papers have been kept as foundations, while some broad surveys and less central prompting papers have been demoted.

## priority A: core papers to read carefully

### RAG foundations

| paper | why it matters |
| --- | --- |
| Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | The foundational RAG paper; use it to define parametric vs non-parametric memory and the baseline motivation for RAG. |
| Karpukhin et al. (2020), [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | Core dense retrieval paper; useful for explaining dense retrieval in QA and why retrieval quality matters. |
| Izacard and Grave (2020), [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) | Important retrieve-then-generate design; useful for top-k context experiments and generator-side evidence aggregation. |
| Gao et al. (2023), [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | Good modern RAG taxonomy: naive RAG, advanced RAG, modular RAG. |
| Huang and Huang (2024), [A Survey on Retrieval-Augmented Text Generation for Large Language Models](https://arxiv.org/abs/2404.10981) | Helpful retrieval-oriented survey; good for pre-retrieval, retrieval, post-retrieval, and generation stages. |
| Gan et al. (2025), [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2504.14891) | Recent RAG evaluation survey; useful for framing retrieval, generation, factuality, safety, and efficiency metrics. |
| Es et al. (2023), [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) | Directly relevant to RAGAS-style faithfulness, answer relevance, context precision, and context recall evaluation. |

### Medical QA, medical RAG, and medical LLMs

| paper | why it matters |
| --- | --- |
| Singhal et al. (2022), [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138) | Key medical LLM evaluation paper; introduces MultiMedQA and human evaluation axes. |
| Jin et al. (2019), [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146) | Classic biomedical QA dataset with question, context, long answer, and short answer. |
| Jin et al. (2020), [What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams](https://arxiv.org/abs/2009.13081) | MedQA paper; useful for situating medical exam QA and open-domain medical retrieval. |
| Pal et al. (2022), [MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering](https://arxiv.org/abs/2203.14371) | Medical multiple-choice QA dataset with explanations; useful parallel to CasiMedicos. |
| Sviridova et al. (2024), [CasiMedicos-Arg: A Medical Question Answering Dataset Annotated with Explanatory Argumentative Structures](https://arxiv.org/abs/2410.05235) | Directly relevant because the thesis uses the CasiMedicos-Arg dataset. |
| Amugongo et al. (2025), [Retrieval augmented generation for large language models in healthcare: A systematic review](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877) | Healthcare-specific RAG review; useful for clinical motivation, risks, evaluation practices, and limitations. |
| Yang et al. (2025), [Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations](https://arxiv.org/abs/2511.05901) | Recent medical RAG scoping review; useful for implementation patterns, clinical applications, evaluation gaps, and ethical issues. |
| Kim et al. (2025), [Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights](https://arxiv.org/abs/2511.06738) | Important cautionary paper: standard medical RAG can degrade performance when retrieval and evidence selection are weak. |
| Zhu (2026), [MRAG: Benchmarking Retrieval-Augmented Generation for Bio-medicine](https://arxiv.org/abs/2601.16503) | Recent medical RAG benchmark; useful for discussing systematic evaluation of biomedical RAG components across tasks and languages. |
| Sivakumar et al. (2026), [RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2603.03541) | Very relevant to error analysis; separates retrieval and generation failures and warns against accuracy-only evaluation in medical RAG. |
| Zhang et al. (2026), [Retrieval-Augmented Generation for Medical Question Answering on a Heart Failure Dataset: Performance Analysis](https://formative.jmir.org/2026/1/e84932) | Recent applied medical QA paper; useful for query taxonomy, risk-aware answer handling, and medical RAG design choices. |

### Evaluation metrics

| paper | why it matters |
| --- | --- |
| Zhang et al. (2019), [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) | Justifies BERT-F1 as a semantic similarity metric. |
| Lin (2004), [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) | Standard lexical/sequence-overlap metric; useful for evidence-style generation. |
| Papineni et al. (2002), [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) | Classic n-gram precision metric; read mainly to explain why BLEU is less ideal for this thesis. |
| Xu et al. (2025), [Does Context Matter? ContextualJudgeBench for Evaluating LLM-based Judges in Contextual Settings](https://arxiv.org/abs/2503.15620) | Useful if you include LLM-as-judge evaluation; highlights that contextual/RAG evaluation is difficult even for judge models. |
| Ju et al. (2025), [Controlled Retrieval-augmented Context Evaluation for Long-form RAG](https://arxiv.org/abs/2506.20051) | Useful for evaluating retrieved context quality separately from final answer quality. |
| Sivakumar et al. (2026), [RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2603.03541) | Also belongs here because it motivates component-level medical RAG evaluation rather than end-to-end accuracy only. |

### Reasoning and self-feedback

| paper | why it matters |
| --- | --- |
| Wei et al. (2022), [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Main reference for visible reasoning / chain-of-thought prompting. |
| Wang et al. (2022), [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | Useful if the thesis discusses multiple reasoning samples or answer aggregation. |
| Madaan et al. (2023), [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | Main reference for a self-feedback pass. |
| Shinn et al. (2023), [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | Useful for agentic self-critique and verbal feedback framing. |
| Yao et al. (2022), [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Relevant if the thesis discusses retrieval as an action or the supervisor's later "agentic reasoner" idea. |
| Xiong et al. (2024/2025), [Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions](https://arxiv.org/abs/2408.00727) | Bridges medical RAG and iterative reasoning; useful if the thesis discusses multi-step retrieval or self-reflection in medical QA. |

## priority B: methods and related work

### Retrieval, embeddings, reranking, and implementation

| paper | why it matters |
| --- | --- |
| Reimers and Gurevych (2019), [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | Background for sentence-transformer retrieval and cosine similarity. |
| Wang et al. (2024), [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672) | Directly relevant to the multilingual-e5-large index. |
| Feng et al. (2020), [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852) | Useful multilingual embedding reference, especially for Spanish/Basque context. |
| Nogueira and Cho (2019), [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) | Basic neural reranking reference; useful if notebook-style reranking is implemented. |
| Khattab and Zaharia (2020), [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) | Strong retrieval/reranking architecture; useful for discussion, not necessarily implementation. |
| Borgeaud et al. (2021), [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426) | RETRO; useful for broader retrieval-augmented language modeling context. |
| Ram et al. (2023), [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083) | Relevant to retrieval inside prompting without finetuning. |
| Stuhlmann et al. (2025/2026), [Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation](https://arxiv.org/abs/2505.07917) | Useful for practical biomedical RAG implementation: BM25, BioBERT, MedCPT, FAISS/Elasticsearch, retrieval depth, latency, and reproducibility. |
| Aljohani et al. (2026), [Enhancing Medical Question Answering with LLMs via a Modular Retrieval-Augmented Generation Framework](https://www.mdpi.com/2078-2489/17/2/133) | Useful if the methods chapter compares sparse, dense, and hybrid retrieval in medical QA. |

### RAG surveys and evaluation surveys

| paper | why it matters |
| --- | --- |
| Zhao et al. (2024), [Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473) | Broad RAG survey; useful for taxonomy and future work, but less central than Gao/Huang/Gan. |
| Gupta et al. (2024), [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions](https://arxiv.org/abs/2410.12837) | Another survey to triangulate terminology and limitations; read selectively. |
| Gan et al. (2025), [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2504.14891) | Focused on evaluation; useful for the metrics chapter and for justifying component-level RAG evaluation. |
| Xu et al. (2025), [mmRAG: A Modular Benchmark for Retrieval-Augmented Generation over Text, Tables, and Knowledge Graphs](https://arxiv.org/abs/2505.11180) | Optional but useful for modular evaluation logic; not medical-specific, but relevant to granular RAG component assessment. |

### Medical RAG and healthcare reliability

| paper | why it matters |
| --- | --- |
| Amugongo et al. (2025), [Retrieval augmented generation for large language models in healthcare: A systematic review](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877) | Healthcare-specific RAG review; use to avoid overclaiming and to frame clinical limitations. |
| Yang et al. (2025), [Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations](https://arxiv.org/abs/2511.05901) | Good for implementation/evaluation patterns in medical RAG and for ethical considerations. |
| Kim et al. (2025), [Rethinking Retrieval-Augmented Generation for Medicine](https://arxiv.org/abs/2511.06738) | Important cautionary paper: retrieval and evidence selection are often the bottlenecks. |
| Dobreva et al. (2025), [RAGCare-QA: A benchmark dataset for evaluating retrieval-augmented generation pipelines in theoretical medical knowledge](https://pmc.ncbi.nlm.nih.gov/articles/PMC12553001/) | Useful benchmark paper if the thesis discusses medical education QA and RAG pipeline complexity. |
| Zhang et al. (2025), [Leveraging Long Context in Retrieval Augmented Language Models for Medical Question Answering](https://www.nature.com/articles/s41746-025-01661-4) | Useful if the thesis discusses long-context models vs retrieval depth and whether larger contexts reduce the need for RAG. |
| Miao et al. (2025), [Improving Large Language Model Applications in Biomedicine With Retrieval-Augmented Generation: A Systematic Review, Meta-Analysis, and Clinical Development Guidelines](https://www.jmir.org/2025/1/e80557) | Useful for clinical-development framing and for retrieval design beyond generic semantic matching. |
| Shah et al. (2023), [Creation and Adoption of Large Language Models in Medicine](https://jamanetwork.com/journals/jama/fullarticle/2808296) | Useful clinical deployment and evaluation framing. |

### Prompting and inference-time methods

| paper | why it matters |
| --- | --- |
| Kojima et al. (2022), [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | Reference for simple "think step by step" style prompting. |
| Zhou et al. (2022), [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625) | Useful if the thesis discusses decomposing clinical questions. |
| Yao et al. (2023), [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | Useful background for deliberate reasoning, but probably not central. |
| Saunders et al. (2022), [Self-critiquing models for assisting human evaluators](https://arxiv.org/abs/2206.05802) | Related to critique/verifier designs. |
| Jeong et al. (2024), [Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i119/7700884) | Highly relevant if self-feedback is applied specifically in medical RAG. |
| Liu et al. (2025), [Judge as A Judge: Improving the Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2502.18817) | Optional; useful if using or discussing LLM-as-judge reliability for RAG evaluation. |

## priority C: models, multilinguality, and thesis extensions

### Open LLMs used or likely to be compared

| paper | why it matters |
| --- | --- |
| Jiang et al. (2023), [Mistral 7B](https://arxiv.org/abs/2310.06825) | Directly relevant if Mistral 7B Instruct is the current baseline model. |
| Touvron et al. (2023), [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | Useful reference for open chat model baselines, but now somewhat dated. |
| Dubey et al. (2024), [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | Better current reference than Llama 2 if Llama 3/3.1 models are compared. |
| Yang et al. (2025), [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | Relevant to think vs no-think comparisons, because Qwen3 explicitly supports thinking/non-thinking modes. |
| Team Gemma (2025), [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) | Relevant if Gemma is included as a secondary model family; useful for multilingual and long-context discussion. |
| Guo et al. (2025), [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | Optional but useful for reasoning-model background if the thesis discusses explicit reasoning behaviour or cost/quality trade-offs. |

### Spanish, Basque, and medical multilingual work

| paper | why it matters |
| --- | --- |
| Etxaniz et al. (2024), [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | Directly relevant to the Basque extension. |
| Domingo-Aldama et al. (2026), [To Adapt or not to Adapt, Rethinking the Value of Medical Knowledge-Aware Large Language Models](https://arxiv.org/abs/2604.06854) | Very relevant to Spanish medical LLMs and Marmoka; useful for comparing medical-adapted vs general LLMs. |
| Agerri et al. / HiTZ resources, [HiTZ Hugging Face models and datasets](https://huggingface.co/HiTZ) | Practical source for CasiMedicos, Latxa-related resources, and medical translation models. |
| Wang et al. (2024), [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672) | Also relevant here because multilingual embedding quality is central to Spanish/Basque retrieval. |
| Yang et al. (2025), [Retrieval-Augmented Generation in Medicine: A Scoping Review](https://arxiv.org/abs/2511.05901) | Include in the multilingual discussion because it explicitly notes gaps in cross-linguistic and low-resource medical RAG. |

## priority D: optional background

| paper | why it matters |
| --- | --- |
| Hendrycks et al. (2020), [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) | Useful background for general LLM benchmark culture. |
| Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | Background for instruction-following models. |
| Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Classic few-shot prompting paper; useful for few-shot format-learning experiments. |
| Nakano et al. (2021), [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) | Useful for retrieval + citation + answer generation background. |
| Filice et al. (2025), [Generating Q&A Benchmarks for RAG Evaluation in Enterprise Settings](https://aclanthology.org/2025.acl-industry.33/) | Optional; useful only if synthetic QA generation is discussed as a future evaluation strategy. |
| Xu et al. (2025), [Towards Global Retrieval Augmented Generation](https://arxiv.org/abs/2510.26205) | Optional; useful only if the thesis discusses corpus-level rather than question-level RAG. |

## suggested reading order

1. RAG foundations: Lewis, DPR, FiD.
2. RAG evaluation: RAGAS, Gan et al. (2025), RAG-X.
3. Medical QA: PubMedQA, MedQA, MedMCQA, CasiMedicos-Arg, MultiMedQA.
4. Medical RAG: Amugongo et al. (2025), Yang et al. (2025), Kim et al. (2025), MRAG, Zhang et al. (2026).
5. Retrieval implementation: SBERT, multilingual E5, reranking, Stuhlmann et al. (2025/2026).
6. Self-feedback and reasoning: Chain-of-thought, Self-Refine, Reflexion, ReAct, Jeong et al. (2024).
7. Model and multilingual extension: Qwen3, Gemma 3, Latxa, Marmoka.
8. Optional background only when writing the introduction or future work.

## notes for the thesis argument

- Use RAG papers to justify the no_rag vs rag ablation.
- Use DPR, FiD, SBERT, E5, and reranking papers to explain the retrieval side of the pipeline.
- Use RAGAS, Gan et al. (2025), RAG-X, and MRAG to justify component-level evaluation rather than answer-only scoring.
- Use Kim et al. (2025) and Yang et al. (2025) to avoid overclaiming: in medicine, better retrieval does not automatically mean better clinical answers.
- Use Self-Refine, Reflexion, and Jeong et al. (2024) to justify self-feedback as an inference-time method, not a separate trained model.
- Use Chain-of-Thought, Qwen3, and DeepSeek-R1 to justify think vs no-think as a cost/quality ablation.
- Use BERTScore, ROUGE, RAGAS, and token overlap to justify measuring both semantic and lexical quality.
- Use CasiMedicos-Arg, Marmoka, Latxa, and multilingual E5 to motivate the Spanish/Basque and multilingual medical QA angle.
- Use RAG-X and Kim et al. (2025) to structure the error analysis around retrieval failure, evidence selection failure, and generation failure.