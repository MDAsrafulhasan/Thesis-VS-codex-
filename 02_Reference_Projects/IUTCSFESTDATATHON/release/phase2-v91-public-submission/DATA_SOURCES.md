# External data and evidence sources

Only `dataset samples.json`, `test set.csv`, and `sample submission.csv` were supplied by the competition. Everything below was external and used for lineage auditing, retrieval, controlled-pair generation, validation, or deterministic evidence. “Used” does not mean that every record was treated as ground truth.

| Source | Link | License / review status | Role |
|---|---|---|---|
| BenHalluEval | [Paper](https://arxiv.org/abs/2605.31483) · [anonymous release](https://anonymous.4open.science/r/BanglaHalluEval-EB77) | No license declaration confirmed in the accessed release; redistribution status pending | Released-lineage and methodology audit |
| IndicXNLI Bengali | [Hugging Face](https://huggingface.co/datasets/Divyanshu/indicxnli) | CC0 | Bengali NLI coverage |
| Bengali Wikipedia dump | [Hugging Face](https://huggingface.co/datasets/rejauldu/bengali-wikipedia) | CC BY 3.0 | Document evidence and hard-pair construction |
| Bengali Wikipedia | [bn.wikipedia.org](https://bn.wikipedia.org/) | CC BY-SA | Exact fact pages and multi-source consensus |
| Bangla TextBook | [Hugging Face](https://huggingface.co/datasets/md-nishat-008/Bangla-TextBook) | MIT | Textbook-domain evidence |
| Bangla newspaper dataset | [Hugging Face](https://huggingface.co/datasets/zabir-nabil/bangla_newspaper_dataset) | MIT | Newspaper-domain evidence |
| Bangladesh Legal Acts | [Hugging Face](https://huggingface.co/datasets/sakhadib/Bangladesh-Legal-Acts-Dataset) | CC BY 4.0 | 35,630 sections normalized into 39,499 evidence chunks |
| Bangla law Q&A | [Hugging Face](https://huggingface.co/datasets/bipinsaha/bangla-law-qna) | CC BY-NC 4.0 | Low-weight auxiliary law evidence |
| Bangladesh Laws portal | [bdlaws.minlaw.gov.bd](http://bdlaws.minlaw.gov.bd/) | Government documents; verify per document | Authoritative section-level evidence |
| Suman Bengali QA | [Hugging Face](https://huggingface.co/datasets/SumanMondal/bengali_qa) | Apache-2.0 | Exact answers, swaps, and context-contrastive pairs |
| Aya Dataset Indic | [Hugging Face](https://huggingface.co/datasets/Cognitive-Lab/Aya_Dataset_Indic) | No license declared; review required | Exact Bengali QA lineage audit |
| BUFFET Bengali TyDiQA-QG | [Hugging Face](https://huggingface.co/datasets/BuffetFS/BUFFET) | MIT | Exact context-question-answer lineage |
| TyDiQA-GoldP Bengali | [Hugging Face](https://huggingface.co/datasets/google-research-datasets/tydiqa) | Apache-2.0 | 2,503 Bengali source rows; strict gold matching |
| SQuAD-BN | [Hugging Face](https://huggingface.co/datasets/csebuetnlp/squad_bn) | CC BY-NC-SA 4.0 | Strict lineage audit |
| BanglaRQA | [Hugging Face](https://huggingface.co/datasets/sartajekram/BanglaRQA) | CC BY-NC-SA 4.0 | Exact overlap audit |
| IndicQA Bengali | [Hugging Face](https://huggingface.co/datasets/ai4bharat/IndicQA) | CC BY 4.0 | Exact overlap audit |
| Bangla question-answer pairs | [Hugging Face](https://huggingface.co/datasets/rasheduzzaman/Bangla_question_answer_pair_70K_dataset) | MIT | Exact overlap audit |
| Participant synthetic hallucination data | [Kaggle](https://www.kaggle.com/datasets/takitajwar17/bengali-hallucination-curated-synthetic) | CC0-1.0 | Audit-only candidate surfacing; never treated as truth |
| Bengali QA from combined data | [Hugging Face](https://huggingface.co/datasets/r-niloy/bengali-qa-from-alldata) | License not declared | Auxiliary retrieval; redistribution review required |
| Bengali QA Dataset | [Hugging Face](https://huggingface.co/datasets/Kowshik24/BengaliQADataset) | License not declared | Auxiliary retrieval |
| Bonolota Bengali QA | [Hugging Face](https://huggingface.co/datasets/Lavlu118557/bonolota_qa_bengali) | License not declared | Literature QA evidence |
| Bangla literature QA | [Hugging Face](https://huggingface.co/datasets/Badhon/bangla_literature_qa_dataset_v_1) | License not declared | Literature QA evidence |
| Pre-independence Bengali QA | [Hugging Face](https://huggingface.co/datasets/sabbirRamim/bengali-question-answering-preindependence-1947-1971) | MIT | Bangladesh-history evidence |
| Bangla MMLU | [Hugging Face](https://huggingface.co/datasets/hishab/bangla-mmlu) | Apache-2.0 in local manifest | Exact MCQ answers |
| Bangla BCS Questions | [Hugging Face](https://huggingface.co/datasets/azminetoushikwasi/bangla-bcs-qs) | Apache-2.0 | Audit-only exact and conservative fuzzy MCQ lineage |
| BCS 10th–40th GK/ICT | [Hugging Face](https://huggingface.co/datasets/azminetoushikwasi/bcs-10-40th-GK-ICT-DM-NMS) | Apache-2.0 | Audit-only exact and conservative fuzzy MCQ lineage |
| Olikbochon Answer Banks bundle | [Kaggle](https://www.kaggle.com/datasets/dinmuhammadrezwoan/olikbochon-answer-banks) | CC BY-SA 4.0 collection; upstream licenses retained | Offline audit bundle for the two BCS sources above and already-listed QA/idiom sources |
| SOMAJGYAAN | [Hugging Face](https://huggingface.co/datasets/farihashifa/SOMAJGYAAN) | MIT | Social/general-knowledge answers |
| BLUCK | [GitHub](https://github.com/minhaj1403/bluck) | CC BY-SA 4.0 | Bengali linguistic and MCQ evidence |
| Bangla Bagdhara | [Kaggle](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara) | CC BY-SA 4.0 | Idiom meanings and controlled pairs |
| Bangla Word Analogy | [GitHub](https://github.com/Mousumi44/Word-Analogy-Bangla) | No repository license declared; audit only, do not redistribute | EMNLP 2023 antonym-pair coverage audit |
| BengaliDictionary | [GitHub](https://github.com/MinhasKamal/BengaliDictionary) | GPL-3.0 | Lexical meanings |
| 80-20 exam bank | [GitHub](https://github.com/tanvirmahfuz100/80-20-exam) | No declared data license | Conservative exam-answer evidence |
| Farhan MCQ API data | [GitHub](https://github.com/mojnupust/api.farhan-mcq) | MIT | Exact Bengali MCQ evidence |
| Job MCQ | [GitHub](https://github.com/NewtonGain/job_mcq) | No declared data license | Low-trust auxiliary answers |
| QDoc | [GitHub](https://github.com/Nh-em0n/QDoc) | No declared data license | Exact question-answer documents |
| Wikidata Query Service | [query.wikidata.org](https://query.wikidata.org/) | CC0-1.0 | Bangladesh facts and controlled swaps |
| Verified fact pages | [`resources/verified_fact_answers.json`](../resources/verified_fact_answers.json) | Mixed public sources | Small bank requiring at least two recorded sources |

Several public Bengali antonym sites and Bengali Wiktionary were also used for multi-source lexical corroboration. Their exact URLs and source relations are recorded in the evidence artifacts and historical README. The scanned *বিপরীতার্থক শব্দের অভিধান* was acquired but not counted as parsed decision evidence because its scan lacked a usable Bengali text layer.

The disabled `shayekh/bengali-exams-public` and `shayekh/bengali-exams` acquisition candidates were not included in the reported corpus and are not counted above.
