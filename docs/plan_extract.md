# Plan Extract

The tentative thesis plan in `plan.pdf` identifies these immediate tasks:

- Test a baseline LLM-only setup with Mistral 7B Instruct.
- Split `SNS1064` into train/dev/test.
- Create and normalize a CasiMedicos dataset with question, options, correct answer, and evidence.
- Format both datasets into a shared schema: question, short answer, evidence.
- Run Spanish dev experiments with:
  - LLM-only baseline.
  - Three-shot prompting using random in-domain retrieval/examples to learn the output format.
  - Specialized retrieval, cross-domain retrieval, and data augmentation variants.
- Later compare LLM-only vs. agentic reasoner on test, and potentially translate/evaluate in Basque.
