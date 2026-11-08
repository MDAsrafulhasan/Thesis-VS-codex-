# Experiment records

Use one subdirectory per actual registered experiment, with the same run ID under `outputs/` and `results/`. Start its `README.md` from [TEMPLATE.md](TEMPLATE.md) and complete the protocol before examining evaluation outcomes.

Keep adaptations, prompt files, approved type mappings, exact commands, environment exports, overlap audits, decisions and failure analysis here. Shared stable modules belong in the active pipeline after their behavior is tested. Do not put new work inside the preserved reference repositories.

Explicitly distinguish:

1. **Synthetic smoke tests:** engineering checks, not empirical thesis evidence.
2. **Independent evaluation:** no benchmark-derived answer/label exposure or source-overlapping calibration.
3. **Lineage-/lookup-assisted reproduction:** legitimate reproduction work, but reported separately rather than claimed as independent generalization.

No empirical experiment records have been fabricated during organization. See [the workflow](../docs/WORKFLOW.md).
