# Citations and licenses

## Competition

অলীকবচন: Bengali LLM Hallucination Detection Challenge. IUT 12th ICT FEST Datathon, Kaggle, 2026.

## Model and inference runtime

- Qwen Team. **Qwen3 model family and technical documentation.** The frozen inference asset is the public Kaggle model [`snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1`](https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1), licensed MIT by its Kaggle publisher.
- ggml-org. **llama.cpp.** Frozen release `b9637`, commit `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`, MIT License. [Upstream repository](https://github.com/ggml-org/llama.cpp).

## Benchmark lineage and language resources

- **BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali.** Used for benchmark-methodology and exact released-lineage auditing.
- **When Words Don’t Mean What They Say: Figurative Understanding in Bengali Idioms.** LREC 2026. Used as organizer-identified background for Bengali figurative-language cases.
- Clark, Jonathan H., et al. **TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages.** Transactions of the Association for Computational Linguistics 8 (2020): 454–470. [Dataset](https://huggingface.co/datasets/google-research-datasets/tydiqa), Apache-2.0.
- [SumanMondal/bengali_qa](https://huggingface.co/datasets/SumanMondal/bengali_qa), Apache-2.0. Used for exact Bengali QA lineage and controlled evidence.
- [BuffetFS/BUFFET](https://huggingface.co/datasets/BuffetFS/BUFFET), MIT. Used for strict Bengali TyDiQA-QG lineage auditing.
- [sakhadib/bangla-bagdhara](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara), CC BY-SA 4.0. Used for exact Bengali idiom evidence.
- [sakhadib/Bangladesh-Legal-Acts-Dataset](https://huggingface.co/datasets/sakhadib/Bangladesh-Legal-Acts-Dataset), CC BY 4.0. Used for normalized section-level Bangladesh-law evidence.
- [Bangladesh Laws](http://bdlaws.minlaw.gov.bd/). Public government legal documents used for authoritative section verification.
- [Wikidata Query Service](https://query.wikidata.org/), CC0. Used for structured Bangladesh-specific fact corroboration.

Additional answer-bank and verified-fact sources are recorded with their URLs, licenses, and trust roles in the private training/audit repository. No unlicensed source is represented here as authoritative ground truth, and no paid or closed inference API is used.
