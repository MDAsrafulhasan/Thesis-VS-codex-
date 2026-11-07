# Citations and licenses

## Competition

অলীকবচন: Bengali LLM Hallucination Detection Challenge. IUT 12th ICT FEST
Datathon, Kaggle, 2026.

Competition Organizers. **Olikbochon Phase 2 Results.** Official final
standings released July 23, 2026. Huntrix score: `0.90144`; rank: `7/32`.

## Model and inference runtime

- Qwen Team. **Qwen3 model family and technical documentation.** The frozen
  inference asset is the public Kaggle model
  [`snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1`](https://www.kaggle.com/models/snowrab/qwen3-6-27b-q4-k-p/pyTorch/default/1),
  whose Kaggle publisher labels the quantization MIT. The official
  [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
  base is Apache-2.0.
- ggml-org. **llama.cpp.** Frozen release `b9637`, commit
  `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`, MIT License.
  [Upstream repository](https://github.com/ggml-org/llama.cpp).

## Benchmark lineage and language resources

- **BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large
  Language Models on Bengali.** Used for benchmark-methodology and exact
  released-lineage auditing.
- **When Words Don’t Mean What They Say: Figurative Understanding in Bengali
  Idioms.** LREC 2026. Used as organizer-identified background and for typed
  literal-versus-figurative routing.
- Clark, Jonathan H., et al. **TyDi QA: A Benchmark for Information-Seeking
  Question Answering in Typologically Diverse Languages.** TACL 8 (2020):
  454–470. [Dataset](https://huggingface.co/datasets/google-research-datasets/tydiqa),
  Apache-2.0.
- [SumanMondal/bengali_qa](https://huggingface.co/datasets/SumanMondal/bengali_qa),
  Apache-2.0.
- [BuffetFS/BUFFET](https://huggingface.co/datasets/BuffetFS/BUFFET), MIT.
- [Bangla Bagdhara](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara),
  CC BY-SA 4.0.
- [Bangladesh Legal Acts](https://huggingface.co/datasets/sakhadib/Bangladesh-Legal-Acts-Dataset),
  CC BY 4.0.
- [Bangladesh Laws](http://bdlaws.minlaw.gov.bd/), public government legal
  documents.
- [Wikidata Query Service](https://query.wikidata.org/), CC0.

The full source register, including audit-only resources with no declared
redistribution license, is provided in `DATA_SOURCES.md`. No paid or closed
inference API is used.
