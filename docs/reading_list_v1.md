# reading list

This is the working literature list for the thesis. Read the priority A papers first, then use priority B and C to fill the related work, methods, and discussion sections.

## priority A: core papers to read carefully

### RAG foundations

| paper | why it matters |
| --- | --- |
| Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | The foundational RAG paper; use it to define parametric vs non-parametric memory and the baseline motivation for RAG. |
| Karpukhin et al. (2020), [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | Core dense retrieval paper; useful for explaining why embeddings can beat sparse lexical retrieval in QA. |
| Izacard and Grave (2020), [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) | Important retrieve-then-generate design; useful for top-k context experiments. |
| Gao et al. (2023), [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | Good modern RAG taxonomy: naive RAG, advanced RAG, modular RAG. |
| Huang and Huang (2024), [A Survey on Retrieval-Augmented Text Generation for Large Language Models](https://arxiv.org/abs/2404.10981) | Helpful retrieval-oriented survey; good for pre-retrieval, retrieval, post-retrieval, and generation stages. |
| Es et al. (2023), [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) | Directly relevant to our RAGAS metrics and faithfulness/context evaluation discussion. |

### Medical QA and medical LLMs

| paper | why it matters |
| --- | --- |
| Singhal et al. (2022), [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138) | Key medical LLM evaluation paper; introduces MultiMedQA and human evaluation axes. |
| Jin et al. (2019), [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146) | Classic biomedical QA dataset with question, context, long answer, and short answer. |
| Jin et al. (2020), [What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams](https://arxiv.org/abs/2009.13081) | MedQA paper; useful for situating medical exam QA and open-domain medical retrieval. |
| Pal et al. (2022), [MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering](https://arxiv.org/abs/2203.14371) | Medical multiple-choice QA dataset with explanations; useful parallel to CasiMedicos. |
| Sviridova et al. (2024), [CasiMedicos-Arg: A Medical Question Answering Dataset Annotated with Explanatory Argumentative Structures](https://arxiv.org/abs/2410.05235) | Directly relevant because we use the CasiMedicos-Arg dataset. |

### Evaluation metrics

| paper | why it matters |
| --- | --- |
| Zhang et al. (2019), [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) | Justifies BERT-F1 as a semantic similarity metric. |
| Lin (2004), [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) | Standard lexical/sequence-overlap metric; useful for evidence-style generation. |
| Papineni et al. (2002), [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) | Classic n-gram precision metric; read mainly to understand why BLEU is less ideal for this thesis. |

### Reasoning and self-feedback

| paper | why it matters |
| --- | --- |
| Wei et al. (2022), [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Main reference for visible reasoning / chain-of-thought prompting. |
| Wang et al. (2022), [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | Useful if we discuss multiple reasoning samples or answer aggregation. |
| Madaan et al. (2023), [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | Main reference for our self-feedback pass. |
| Shinn et al. (2023), [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | Useful for agentic self-critique and verbal feedback framing. |
| Yao et al. (2022), [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Relevant for the supervisor's later "agentic reasoner" idea. |

## priority B: methods and related work

### Retrieval, embeddings, and reranking

| paper | why it matters |
| --- | --- |
| Reimers and Gurevych (2019), [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | Background for sentence-transformer retrieval and cosine similarity. |
| Wang et al. (2024), [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672) | Directly relevant to the multilingual-e5-large index. |
| Feng et al. (2020), [Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852) | Useful multilingual embedding reference, especially for Spanish/Basque context. |
| Nogueira and Cho (2019), [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) | Basic neural reranking reference; useful if we implement notebook-style reranking. |
| Khattab and Zaharia (2020), [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) | Strong retrieval/reranking architecture; useful for discussion, not necessarily implementation. |
| Borgeaud et al. (2021), [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426) | RETRO; useful for broader retrieval-augmented language modeling context. |
| Ram et al. (2023), [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083) | Relevant to retrieval inside prompting without finetuning. |

### RAG surveys and evaluation surveys

| paper | why it matters |
| --- | --- |
| Zhao et al. (2024), [Retrieval-Augmented Generation for AI-Generated Content: A Survey](https://arxiv.org/abs/2402.19473) | Broad RAG survey; useful for taxonomy and future work. |
| Gupta et al. (2024), [A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions](https://arxiv.org/abs/2410.12837) | Another survey to triangulate terminology and limitations. |
| Gan et al. (2025), [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2504.14891) | Focused on evaluation; useful for the metrics chapter. |

### Medical RAG and healthcare reliability

| paper | why it matters |
| --- | --- |
| Amugongo et al. (2025), [Retrieval augmented generation for large language models in healthcare: A systematic review](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877) | Healthcare-specific RAG review; useful for clinical motivation and limitations. |
| Yang et al. (2025), [Retrieval-Augmented Generation in Medicine: A Scoping Review of Technical Implementations, Clinical Applications, and Ethical Considerations](https://arxiv.org/abs/2511.05901) | Good for implementation/evaluation patterns in medical RAG. |
| Kim et al. (2025), [Rethinking Retrieval-Augmented Generation for Medicine](https://arxiv.org/abs/2511.06738) | Important cautionary paper: medical RAG can hurt when retrieval/evidence selection is weak. |
| Shah et al. (2023), [Creation and Adoption of Large Language Models in Medicine](https://jamanetwork.com/journals/jama/fullarticle/2808296) | Useful clinical deployment and evaluation framing. |

### Prompting and inference-time methods

| paper | why it matters |
| --- | --- |
| Kojima et al. (2022), [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | Reference for simple "think step by step" style prompting. |
| Zhou et al. (2022), [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625) | Useful if we discuss decomposing clinical questions. |
| Yao et al. (2023), [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | Useful background for deliberate reasoning, but probably not central. |
| Saunders et al. (2022), [Self-critiquing models for assisting human evaluators](https://arxiv.org/abs/2206.05802) | Related to critique/verifier designs. |

## priority C: models, multilinguality, and thesis extensions

### Open LLMs used or likely to be compared

| paper | why it matters |
| --- | --- |
| Jiang et al. (2023), [Mistral 7B](https://arxiv.org/abs/2310.06825) | Directly relevant because Mistral 7B Instruct is the current baseline model. |
| Touvron et al. (2023), [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | Useful reference for open chat model baselines. |
| Yang et al. (2025), [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | Relevant to think vs no-think comparisons, because Qwen3 explicitly supports thinking/non-thinking modes. |
| Team Gemma (2025), [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) | Relevant if Gemma is included as a secondary model family. |

### Spanish, Basque, and medical multilingual work

| paper | why it matters |
| --- | --- |
| Etxaniz et al. (2024), [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | Directly relevant to the Basque extension. |
| Domingo-Aldama et al. (2026), [To Adapt or not to Adapt, Rethinking the Value of Medical Knowledge-Aware Large Language Models](https://arxiv.org/abs/2604.06854) | Relevant to Spanish medical LLMs and Marmoka; read if we compare medical-adapted vs general LLMs. |
| Agerri et al. / HiTZ resources, [HiTZ Hugging Face models and datasets](https://huggingface.co/HiTZ) | Practical source for CasiMedicos, Latxa-related resources, and medical translation models. |

## priority D: optional background

| paper | why it matters |
| --- | --- |
| Hendrycks et al. (2020), [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) | Useful background for general LLM benchmark culture. |
| Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | Background for instruction-following models. |
| Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Classic few-shot prompting paper; useful for our few-shot format-learning experiments. |
| Nakano et al. (2021), [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) | Useful for retrieval + citation + answer generation background. |

## suggested reading order

1. RAG foundations: Lewis, DPR, FiD.
2. Evaluation: BERTScore, ROUGE, RAGAS.
3. Medical QA: Med-PaLM/MultiMedQA, PubMedQA, MedQA, MedMCQA, CasiMedicos-Arg.
4. Self-feedback and reasoning: Chain-of-thought, Self-Refine, Reflexion, ReAct.
5. Retrieval implementation: SBERT, multilingual E5, reranking.
6. Medical RAG reviews and cautionary papers.
7. Spanish/Basque/model papers when writing the extension chapter.

## notes for the thesis argument

- Use RAG papers to justify the no_rag vs rag ablation.
- Use Self-Refine and Reflexion to justify self-feedback as an inference-time method, not a separate trained model.
- Use Chain-of-Thought and Qwen3 to justify think vs no-think as a cost/quality ablation.
- Use RAGAS, BERTScore, ROUGE, and token overlap to justify measuring both semantic and lexical quality.
- Use medical RAG review papers to avoid overclaiming: in medicine, better retrieval does not automatically mean better clinical answers.
