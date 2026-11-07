# Real dialect benchmark not yet supplied

This directory is reserved for original, licensed dialect data and its provenance documentation. No observations or human ratings have been fabricated.

Required canonical columns: `id,dialect,question_type,question,response,human_or_primary_score`.

Document collection source, license/consent where relevant, dialect validation, shared semantic content, annotation rubric and judge/annotator identities. Scores must be finite in [0,10]; question types are `definitional` and `factual`. The planned anchor subset uses Standard_Bengali, Tangail and Chittagong.

Synthetic fixtures belong under `data/splits/mock_dialect_seed42/`, never here. Their dialect wording and scores are not validated linguistic or human-judgment evidence. See [the data guide](../../../docs/DATA_GUIDE.md).
