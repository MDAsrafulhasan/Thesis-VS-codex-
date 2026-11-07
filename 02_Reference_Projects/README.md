# Reference projects: reading guide

These are complete external datathon research artifacts. They are retained as reference material, with their internal code, data, tests, releases, and supporting documents kept together. Cite and describe their results as prior competition work; do not present their leaderboard scores or validation sets as results from the thesis experiments.

## 1. IUT Datathon Hallu

Path: [`IUT_Datathon_Hallu/`](IUT_Datathon_Hallu/)

This compact project contains the CUET_AL_Masaar report, presentation, inference notebook, README, and license for the IUT Bengali hallucination detection datathon.

Recommended order:

1. [`README.md`](IUT_Datathon_Hallu/README.md) — task, inference instructions, and method overview.
2. [`CUET_AL_Masaar_report.pdf`](IUT_Datathon_Hallu/CUET_AL_Masaar_report.pdf) — team report.
3. [`CUET_AL_Masaar_presentation.pdf`](IUT_Datathon_Hallu/CUET_AL_Masaar_presentation.pdf) — presentation.
4. [`CUET_AL_Masaar_inference_notebook.ipynb`](IUT_Datathon_Hallu/CUET_AL_Masaar_inference_notebook.ipynb) — implementation and inference flow.

## 2. IUT CS Fest Datathon / Huntrix

Path: [`IUTCSFESTDATATHON/`](IUTCSFESTDATATHON/)

This is a larger evidence-first Bengali hallucination detection project. It includes source code, regression tests, data/provenance notes, frozen Kaggle releases, notebooks, a technical report, presentation, and decision/experiment histories.

Recommended order:

1. [`README.md`](IUTCSFESTDATATHON/README.md) — results, architecture, constraints, and project map.
2. [`Problem.md`](IUTCSFESTDATATHON/Problem.md) — original competition task and requirements.
3. [`docs/TECHNICAL_PAPER.md`](IUTCSFESTDATATHON/docs/TECHNICAL_PAPER.md) and [`output/pdf/Huntrix_Technical_Report.pdf`](IUTCSFESTDATATHON/output/pdf/Huntrix_Technical_Report.pdf) — technical explanation.
4. [`docs/DATA_SOURCES.md`](IUTCSFESTDATATHON/docs/DATA_SOURCES.md) and [`docs/COMPLIANCE_REVIEW.md`](IUTCSFESTDATATHON/docs/COMPLIANCE_REVIEW.md) — data provenance and competition compliance.
5. [`DECISION.log`](IUTCSFESTDATATHON/DECISION.log), [`EXPERIMENT.log`](IUTCSFESTDATATHON/EXPERIMENT.log), and [`docs/JOURNEY.md`](IUTCSFESTDATATHON/docs/JOURNEY.md) — modeling decisions, experiment records, and progression.
6. [`release/phase2-v91-public-submission/`](IUTCSFESTDATATHON/release/phase2-v91-public-submission/) and [`kaggle_notebooks/`](IUTCSFESTDATATHON/kaggle_notebooks/) — frozen deliverables and inference/audit notebooks.
7. [`scripts/`](IUTCSFESTDATATHON/scripts/) and [`tests/`](IUTCSFESTDATATHON/tests/) — implementation and regression coverage.

Before changing the Huntrix project’s modeling code, follow its [`AGENTS.md`](IUTCSFESTDATATHON/AGENTS.md): read `Problem.md`, `README.md`, `DECISION.log`, and `EXPERIMENT.log`; preserve the stated offline/open-weight constraints; and record new experiments in its log if the work belongs to that project.

## Integrity

[`REFERENCE_FILE_MANIFEST.csv`](REFERENCE_FILE_MANIFEST.csv) lists the relative path, size, and SHA-256 checksum of every original file in both projects. The two projects’ internal README files remain the primary documentation for their own code and artifacts.
