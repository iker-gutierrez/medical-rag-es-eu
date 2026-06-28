**1\. kimEtAl2025**: Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights

* (LLMs) are transforming the landscape of medicine, yet two fundamental challenges persist: keeping up with rapidly evolving medical knowledge and providing verifiable, evidence-grounded reasoning  
* Contrary to expectation, standard RAG often degraded performance  
* that simple yet effective strategies, including evidence filtering and query reformulation, substantially mitigate these issues, improving performance on MedMCQA and MedXpertQA by up to 12% and 8.2%, respectively   
* first large-scale, fine-grained evaluation of widely used medical RAG frameworks to elucidate their behavior throughout the end-to-end pipeline.  
* evaluated two LLMs (GPT-4o and Llama-3.1-8B) on **200 queries** (real patient inquiries and USMLE-style questions) across evidence retrieval, evidence selection, and response generation  
* **Eighteen medical experts contributed over 80,000 expert annotations**  
* While many prior studies have highlighted RAG’s ability to improve factuality and mitigate hallucinations in medical applications29, 42, 49, 50, others–particularly in the general domain–have reported that irrelevant or noisy retrieved content can distract models or sustain hallucinations51–53. This aligns with our findings: when the model incorporated irrelevant rather than relevant passages (i.e., the “False Positive” category in Fig. 4), statement-level factuality scores declined, with the effect being especially pronounced for the smaller model, Llama-3.1, which exhibited a drop of over 8%.  
* Interestingly, when a relevant passage was retrieved but not cited by the model (i.e., Supported but Missed), completeness still dropped by 1-5%, suggesting that the model may have been distracted by other retrieved passages and failed to integrate critical evidence.  
* the retrieval accuracy for medical queries itself has rarely been evaluated in a standardized, expert-grounded manner. Our manual evaluation addressed this gap, showing that current retrievers frequently mishandled clinical questions.  
* Refs:  
  * 29\. Zakka, C. et al. Almanac—retrieval-augmented language models for clinical medicine. Nejm ai 1, AIoa2300068 (2024).  
  * 42\. Manes, I. et al. K-qa: A real-world medical q\&a benchmark. In Proceedings of the 23rd Workshop on Biomedical Natural Language Processing, 277–294 (2024).  
  * 49\. Miao, J., Thongprayoon, C., Suppadungsuk, S., Garcia Valencia, O. A. & Cheungpasitporn, W. Integrating retrieval-augmented generation with large language models in nephrology: advancing practical applications. Medicina 60, 445 (2024).  
  * 50\. Luo, M.-J. et al. Development and evaluation of a retrieval-augmented large language model framework for ophthalmology. JAMA ophthalmology 142, 798–805 (2024).

**2\. sivakumarEtAl2026**: RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical Question Answering

* Despite progress in RAG evaluation, current benchmarks focus only on simple multiple-choice QA tasks and employ metrics that poorly capture the semantic precision required for complex QA tasks.  
* RAG-X, a diagnostic framework that evaluates the retriever and generator independently across a triad of QA tasks:  
  * information extraction  
  * short-answer generation  
  * multiple-choice question (MCQ) answering.  
* RAG-X introduces Context Utilization Efficiency (CUE) metrics to disaggregate system success into interpretable quadrants, isolating verified grounding from deceptive accuracy.

**3\. LiangEtAl2025**: RGAR: Recurrence Generation-augmented Retrieval for Factual-aware  
Medical Question Answering

* existing retrieval approaches often overlook the patient-specific factual knowledge embedded in Electronic Health Records (EHRs),  
* This paper introduces **RGAR**, a recurrence generation-augmented retrieval framework that synergistically retrieves both factual and conceptual knowledge from dual sources (i.e., EHRs and the corpus),  
* Our findings demonstrate the **benefit of explicitly mining patient-specific factual knowledge** during retrieval, consistently improving generation quality and clinical relevance.  
* Related work:  
  * RAG systems are characterized as a **"Retrieve-then-Read"** framework (Gao et al., 2023\)  
  * The development of **Naive RAG** has primarily focused on retriever optimization, evolving from discrete retrievers such as BM25 (Friedman et al., 1977\) to more sophisticated and domain-specific dense retrievers, including DPR (Karpukhin et al., 2020\) and MedCPT (Jin et al., 2023), which demonstrate superior performance.  
  * **Advanced RAG** systems focus on designing multi-round retrieval structures, including iterative retrieval (Sun et al., 2019), recursive retrieval (Sarthi et al., 2024), and adaptive retrieval  
  * A notable work in medical QA is **MedRAG** (Xiong et al., 2024a), which analyzes retrievers, corpora, and LLMs, offering practical guidelines. Follow-up work, i-MedRAG (Xiong et al., 2024b), improved performance through multi-round decomposition and iteration, albeit with significant computational costs.  
  * **Agentic RAG** frameworks have emerged, treating RAG and CoT as executable actions that can be flexibly invoked **based on an LLM’s metacognitive capabilities**  
  * **Query optimization**, or prompt optimization, is crucial for improving AI system performance, especially for retrieval-based tasks. It is widely applied in fields like text-toimage (Liu et al., 2022; Wu et al., 2024b) and code generation (Nazzal et al., 2024).

**4\. xiongEtAl2024:** Benchmarking Retrieval-Augmented Generation for Medicine

* there is a lack of best practices regarding the optimal RAG setting for various medical purposes. To systematically evaluate such systems, we propose the Medical Information Retrieval-Augmented Generation Evaluation (MIRAGE), a first-of-its-kind benchmark including 7,663 questions from five medical QA datasets.  
* Overall, MEDRAG improves the accuracy of six different LLMs by up to 18% over chain-of-thought prompting, elevating the performance of GPT-3.5 and Mixtral to GPT-4-level.  
* We discovered a log-linear scaling property and the “lost-in-the-middle” effects in medical RAG.  
* Evaluation Settings:  
  * Zero-Shot Learning (ZSL).  
  * Multi-Choice Evaluation (MCE).  
  * ​​Retrieval-Augmented Generation (RAG).  
  * Question-Only Retrieval (QOR). To align with real-world cases of medical QA, answer options should not be provided as input during retrieval.

**5\. ozakiEtAl2025**: Understanding the Impact of Confidence in Retrieval Augmented Generation: A Case Study in the Medical Domain

* We evaluate confidence by treating the model’s predicted probability as its output and calculating several evaluation metrics which include calibration error method, entropy, the best probability, and accuracy. Experimental results across multiple datasets confirmed that certain models possess the capability to judge for themselves whether an inserted document relates to the correct answer.  
* For the dataset, we select PubMedQA (Jin et al., 2019\) and MedMCQA (Pal et al., 2022), both of which include multiple-choices QA data along with explanatory passages that justify the answers. These datasets follow the experimental setup of MIRAGE (Xiong et al., 2024\)  
* Our study modifies the approach based on the MIRAGE paper (Xiong et al., 2024\) by excluding Chain of Thought (CoT) (Wei et al., 2022), allowing direct probability computation. (In other words, when using CoT, the model must generate responses, which, as discussed in Section 3, prevents a valid evaluation.)  
* The results of Phi-3.5 on PubMedQA reveal that the incorrect predictions tend to concentrate at the upper end, i.e., where output probabilities are high. This pattern suggests that the model exhibits overconfidence, making incorrect predictions despite assigning high probabilities. When solving a QA task under a deliberate setting that includes supporting documents for correct answers (similar to pseudo-RAG), all models (Phi and Qwen) showed improved output probabilities. This suggests that the models can assess whether retrieved documents contain useful information.  
* Behavior of Calibration Error. model. Notably, even when inserting entirely correct documents (Ans1), Llama3.1 (70B) experiences a drop in accuracy, whereas Llama3.1 (8B) shows improved accuracy even when inserting completely unrelated documents (Oth3). This stark contrast indicates that even within the same model family, behavior can vary largely. Moreover, a comparison between Llama2, Meditron, and Llama3.1 shows considerable differences in behavior, ruling out parameter size as the primary cause.  
* Our study also examined the “Lost in the Middle” phenomenon (Liu et al., 2024\) by evaluating the impact of document placement within the template across multiple positions (Pre-Q, Aft-Q, and Aft-C).  
* **We found that when models retrieve relevant documents, they not only boost accuracy but also show higher confidence scores**. In contrast, irrelevant documents have little effect on improving confidence. Several models demonstrate the ability to judge if the retrieved documents connect to the correct answer, indicating a more discerning use of external information than we anticipated. Our

**6\. asaiEtAl2023:** Self-Rag: Learning To Retrieve, Generate, And Critique Through Self-Reflection

* Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. Generating  
* Experiments show that SELF-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks.  
* Reflection tokens are categorized into retrieval and critique tokens to indicate the need for retrieval and its generation quality respectively SELF-RAG first determines if augmenting the continued generation with retrieved passages would be helpful. If so, it outputs a retrieval token that calls a retriever model on demand (Step 1). Subsequently, SELF-RAG concurrently processes multiple retrieved passages, evaluating their relevance and then generating corresponding task outputs (Step 2). It then generates critique tokens to criticize its own output and choose best one (Step 3\) in terms of factuality and overall quality.  
* Reflection tokens, inspired by reward models used in reinforcement learning (Ziegler et al., 2019; Ouyang et al., 2022), are inserted offline into the original corpus by a trained critic model. This eliminates the need to host a critic model during training, reducing overhead.  
* In particular, our inference-time algorithm enables us to (1) flexibly adjust retrieval frequency for different downstream applications and (2) customize models’ behaviors to user preferences by leveraging reflection tokens through segment-level beam search using the weighted linear sum of the reflection token probabilities as segment score.  
* Retrieval-Augmented Generation. Retrieval-Augmented Generation (RAG) augments the input space of LMs with retrieved text passages (Guu et al., 2020; Lewis et al., 2020), leading to large improvements in knowledge-intensive tasks after fine-tuning or used with off-the-shelf LMs (Ram et al., 2023).  
* Though our work also studies fine-grained critique on retrieval and generation, we train our target LM on task examples augmented with reflection tokens from a critic model offline, with a far lower training cost compared to RLHF.   
* Other works use general control tokens to guide LM generation (Lu et al., 2022; Korbak et al., 2023), while SELF-RAG uses reflection tokens to decide the need for retrieval and to self-evaluate generation quality. Xie et al. (2023) propose a self-evaluation guided decoding framework, but they focus only on reasoning tasks with one evaluation dimension (reasoning path consistency) and without retrieval.  
* This work introduces SELF-RAG, a new framework to enhance the quality and factuality of LLMs through **retrieval on demand and self-reflection.**  
* ETHICAL CONCERNS. This work aims to improve the factuality of LLM outputs, the lack of which continues to cause numerous real-world problems (e.g., spread of misinformation and provision of incorrect and dangerous advice). While our method shows significant improvements in terms of performance, factuality, and citation accuracy, it can still generate outputs that are not fully supported by the citations. We hope that explicit self-reflection and fine-grained attribution may help users verify factual errors in the model outputs.

**7\. asaiEtAl2026**: Synthesizing scientific literature with retrieval-augmented language models

* we develop ScholarQABench, the \!rst large-scale multi-domain benchmark for literature search  
* model, OpenScholar-8B outperforms GPT-4o by 6.1% and PaperQA2 by 5.5% in correctness on a challenging multi-paper synthesis task from the new ScholarQABench. Whereas **GPT-4o hallucinates citations 78–90% of the time**, OpenScholar can produce high-quality outputs that are not only on par with expert-written answers but, in some cases, above par, particularly in terms of coverage and organization. We also released the first public demo for scientific literature synthesis, powered by OpenScholar-8B. Since launch, the demo has been used by more than 30,000"users and has collected nearly 90,000"user queries across diverse scientific fields.  
* literature across fields such as computer science and biomedicine. Retrieval-augmented LMs5–7 mitigate some of these issues by incorporating external knowledge at inference time and have encouraged systems for literature search and synthesis8–10. However, most rely on black-box application programming interfaces (APIs) or general-purpose LMs and lack open, domain-specific retrieval data stores (processed corpora with retrieval indices) tailored to scientific domains. Evaluations for literature synthesis are also limited, typically focusing on narrow, single-discipline studies8,9 or simplified tasks such as multiple-choice question answering.  
* OpenScholar integrates a domain-specialized data store (OpenScholar DataStore, OSDS), adaptive retrieval modules and a new self-feedback-guided generation mechanism that enables iterative refinement of long-form outputs. accuracy. This same pipeline is used to generate high-quality synthetic data, enabling the training of a compact 8B model (OpenScholar-8B) and retrievers without relying on proprietary LMs.  
* To evaluate OpenScholar, we introduce ScholarQABench (Fig. 1, middle), to our knowledge the first multidisciplinary benchmark for open-ended scientific synthesis.  
* ScholarQABench requires long-form responses grounded in up-to-date literature from numerous papers. It includes 3,000 research questions and 250"expert-written answers across computer science, physics, biomedicine and neuroscience, authored by experienced PhD students and postdocs to reflect real-world literature review practices.  
* Our expert analysis shows that the proposed multifaceted evaluation pipeline achieves high agreement with expert judgements, reliably capturing coverage, coherence, writing quality and factual correctness  
* We compare three settings. (1) Parametric LMs (no retrieval): Llama" 3.1"8B/70B (ref."17) and GPT-4o (gpt-4o-2024-05-13 (ref."18)) generate answers and a list of paper titles. We verify that the titles exist and, when they do, fetch their abstracts as citations. (2) Retrieval-augmented generation (RAG) baselines: using our OSDS (RAGOSDS), we retrieve the top N passages and concatenate them with the input, following standard RAG pipelines2,18. (3) Our method (OpenScholar): a custom inference pipeline with a trained 8B model (OpenScholar-8B) and with Llama"3.1"70B and GPT-4o back ends (OpenScholar-70B, OpenScholar-GPT-4o). For multi-paper tasks, we also test Perplexity Pro. We use the paid subscription version  
* Main results. On single-paper tasks, OpenScholar consistently outperforms other models. In multi-paper tasks, we report the Scholar-CS rubric score—the number of expert-annotated answer rubrics satisfied by the response of a model (see Methods for scoring details)—as our primary measure of correctness. We also evaluate overall writing quality with a LLM judge (‘LLM’) on Scholar-Multi and track citation accuracy across all datasets.  
* Although we found that PaperQA2 matches or even outperforms OpenScholar in citation accuracy, its responses often rely on only one or a few papers, summarizing each retrieved snippet individually. This leads to limited coverage and contributes to its lower performance on the Scholar-CS rubric and LLM judge scores. These findings highlight the importance of balancing both precision and recall in effective literature synthesis.  
* Notably, by making use of efficient retrieval pipelines with lightweight bi-encoders, cross-encoders and in-house models, OpenScholar-8B and OpenScholar-GPT-4o achieve much lower costs—orders of magnitude cheaper than PaperQA2—while maintaining high performance.

