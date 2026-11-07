অলীকবচন | Bengali LLM Hallucination Detection Challenge
Detect hallucinations in Bengali language-model outputs.


অলীকবচন | Bengali LLM Hallucination Detection Challenge

Submit Prediction
Overview
অলীকবচন: Bengali LLM Hallucination Detection Challenge
Datathon 2.0 | powered by the Institute of Policy Dynamics | sponsored by Brain Lab | organised by IUTCS

Goal of the Competition
Build a system that, given a Bengali prompt and a candidate response, predicts whether the response is faithful (label = 1) or hallucinated (label = 0). Some items include a supporting context passage; some do not. Your detector must handle both.

The metric is binary F1 on the hallucinated class (macro F1). The competition runs in two phases: a Kaggle leaderboard phase, followed by an in-person final at Islamic University of Technology (IUT) where the top 15 teams compete on a held-out fold, present their approach, and submit a short paper. Full details are in the Rules and Evaluation tabs.

Why This Competition Exists
Bengali is spoken by over 280 million people. The large language models that increasingly mediate how Bengali speakers access information — through chatbots, search assistants, government services, health information systems, and educational tools — are remarkably fluent in Bengali. They are also, frequently, confidently wrong.

A model can correctly identify the author of Bidrohi when asked in English and then attribute the poem to Rabindranath Tagore when asked in Bengali. It can describe Bangladesh's Liberation War with fluent prose and the wrong dates. It can produce a paragraph about a national monument that reads beautifully and contains a fabricated commissioning year. These failures don't sound wrong. They sound like Bengali — most of the time — which is exactly what makes them dangerous.

অলীকবচন — the mirage of language — asks you to build a system that catches these mirages.

What the Data Looks Like
The benchmark spans multiple domains, three cultural-distance bands from globally-stable facts to deeply Bangladesh-specific knowledge, and four task types from closed-book question answering to translation. Some of the wrong responses are real outputs harvested from frontier and Bengali-native LLMs. Others are theory-motivated examples designed around specific failure modes documented in recent multilingual hallucination research. All of them are the kind of error a Bengali speaker actually encounters in the wild.

Cultural-distance bands are the structure the task actually rewards understanding:

C0 — globally stable facts (universal science, world geography, mathematics). Models should get these right in any language. Control band.
C1 — culturally situated. The Bangladeshi answer differs from a Western or globally-dominant default, or the fact exists only in the Bangladeshi context. This is where the headline phenomenon lives.
C2 — contested, recent, or time-sensitive facts.
Full schema, file descriptions, and label definitions are in the Data tab.

What Makes This a Research Competition
There is no large training set in the conventional sense. A sample split with labels is released so you can validate your pipelines locally; the test set is held out and used for the public and private leaderboards. The goal is not to fine-tune on our examples. The goal is to build a detector that generalizes — one that catches hallucinations whether they came from GPT-4o, Llama, TituLLM, or a model that doesn't exist yet.

Two working starter notebooks will be provided to give you the floor. Keep an eye on the competition Kaggle page!

The top 5 teams by final combined score are invited to co-author the findings paper submitted to a top NLP venue. Your solution becomes evidence in the literature on how cross-lingual hallucination behaves in low-resource languages and how it can be reliably caught. That citation will outlast the medal.

What Makes This a Community Competition
Bengali language technology will not improve unless the community pushes it forward. This competition exists in that lineage. Every submission here, win or lose, becomes part of how we measure and reduce hallucination in Bengali AI systems going forward.

Whether you're a Kaggle veteran, an undergraduate student attempting your first NLP competition, or an international researcher curious about what hallucination looks like outside English, this competition is built for you. If your team has trained its own Bengali model — a fine-tune, a from-scratch effort, an unpublished research artifact — you're welcome to bring it. Rules for using your own models are in the External Data and Models section of the Rules tab. But you do need to register to be able to participate!

Getting Started
Read the Data tab carefully. Browse a few sample-split rows in each cultural-distance band. The difference matters.
Read the Rules tab, especially the Compute and Inference Rules section. This is a code competition — at Phase 2, submissions run offline in a Kaggle kernel with open-weight models only.
Fork one of the starter notebooks and get a submission on the board.
Use the Discussion tab. Ask questions. If something in the data looks wrong, tell us — we'll fix what we get wrong.
Good luck. We can't wait to see what you build.

— The competition organizing team

The Brain Lab



The Brain Lab under the ICT Division of the Government of Bangladesh is the exploratory, research-focused arm of the Enhancement of Bengali Language in ICT Through Research and Development (EBLICT) project. Implemented by the Bangladesh Computer Council, the lab is dedicated to strengthening the Bengali language in digital systems and building the nation's public AI and language infrastructure.Core Research DomainsThe lab’s initiatives are divided into six primary areas:Language Technologies: Developing Bengali NLP, corpora, and grammar systems to support national digital infrastructure.Speech & Multimodal Systems: Creating text-to-speech, speech-to-text, and sign language recognition tools.Accessibility & Inclusive Systems: Designing assistive communication technologies and inclusive digital interfaces.Knowledge & Cultural Systems: Digitally archiving linguistic records and preserving cultural heritage.Human-Centered AI Design: Conducting user experience research and defining ethical AI frameworks.Learning & Technical Education: Producing no-code tools, visual programming interfaces, and entry-level robotics for technical learning.Key National InitiativesAs a premier public-sector R&D unit, Brain Lab and the EBLICT project have been instrumental in launching major digital language platforms, including:Kagoj.ai: An AI-based writing and productivity platform designed for the Bangla language.

Powered by Institute of Policy Dynamics

.svg?generation=1782920598895271&alt=media)
Most policy research is a collection of separate studies on separate topics. The Institute of Policy Dynamics is organised around a single question — and every piece of work is a lens on it.
Several pressures usually treated as Bangladesh’s particular problems — low institutional trust, weak civic participation, and conflict over scarce resources — may be the early form of a condition wealthier, more stable societies are drifting toward. IPD holds this as a hypothesis to test, not a truth to assert.

Start

42 minutes ago
Close

18 days to go
Description
Bengali is spoken by over 280 million people. The large language models that increasingly mediate how Bengali speakers access information through chatbots, search assistants, government services, health information systems, and educational tools are remarkably fluent in Bengali. They are also, frequently, confidently wrong.
A model can correctly identify the author of Bidrohi when asked in English and then attribute the poem to Tagore when asked in Bengali. It can describe Bangladesh's Liberation War with fluent prose and the wrong dates. It can produce a paragraph about any national monument that reads beautifully and contains a fabricated commissioning year. These failures don't sound wrong. They sound like Bengali (most of the times), which is exactly what makes them dangerous.

This competition asks you to build a system that catches these mirages.

Given a Bengali prompt and a response (sometimes with a source passage, sometimes without) your task is to predict whether the response is faithful or hallucinated. The benchmark spans different domains, cultural-distance bands from globally-stable facts to deeply Bangladesh-specific knowledge, and different task types from closed-book question answering to translation. Some of the wrong responses are real outputs harvested from frontier and Bengali-native LLMs. Others are theory-motivated examples designed around the specific failure modes documented in recent multilingual hallucination research. All of them are the kind of error a Bengali speaker actually encounters in the wild.

This is a research competition. The leaderboard ranks you on F1 of the hallucinated class, with the heaviest weight on the culturally-situated band where the headline phenomenon concentrates. There is no training set in the conventional sense — a sample set is released for pipeline validation, and detection systems will be judged on how well they generalize.

This is also a community competition. It exists because Bengali language technology will not improve unless the community pushes it forward. Every submission here, win or lose, becomes part of how we measure and reduce hallucination in Bengali AI systems going forward.

We hope you'll join us.

Evaluation
Submissions are evaluated on macro-F1 score.

Submission File
For each id in the test set, you must predict for the TARGET variable (0= hallucinated and 1 = not hallucinated). The file should contain a header and have the following format:

id,label
1,1
2,1
3,1
etc.
Citation
Ahmed Shafin Ruhan, Arefin_Mizan, Jubaer_36, Shefayat, and Sushmit. অলীকবচন | Bengali LLM Hallucination Detection Challenge. https://kaggle.com/competitions/bengali-hallucination, 2026. Kaggle.


Cite
Competition Host
Sushmit

Prizes & Awards
Kudos

Does not award Points or Medals

Participation
4 Entrants

1 Participant

1 Team

1 Submission

Tags
F-Score (Macro)
Table of Contents





Dataset Description
Bangla Hallucination Detection — Dataset Description
Overview
This competition asks participants to detect hallucinations in Bangla-language LLM responses. Given a context (optional supporting text, or [NULL] if none is provided), a prompt_bn (the question or instruction given to a model), and a response_bn (answer in Bangla which may or may not contain cases of hallucination; both intrinsic or extrinsic), the task is to predict whether the response contains a hallucination.

Files
Sample file: A smaller, hand-curated set of examples with ground-truth annotations included. Use this to get idea about the data, and validate your models.
Columns: context, prompt_bn, response_bn, label
test.csv: A larger set of examples in the same format, but without the label column. Your goal is to predict labels for every row in this file.
sample_submission.csv: Shows the expected submission format — one column with the row/entry number (id), and one column with your predicted label (label).
Label Definition
label = 0 — The response contains a hallucination: it includes information that is fabricated, unsupported by the given context, or factually incorrect.
label = 1 — The response is faithful/non-hallucinated: it is accurate and, where a context is provided, properly grounded in it.
Submission Format
Submit a CSV file with exactly two columns:

id,label
0,1
1,0
2,1
...
Each row corresponds to an entry in test.csv, and label should be your predicted 0 or 1 for that entry.



Competition Rules
Rules
By entering this competition you agree to the following rules. Read them carefully. Ambiguity or disputes will be resolved at the discretion of the organizing team.

1. Team Composition
Teams may have between 1 and 4 members.
Each participant may be a member of only one team for the duration of the competition. Multiple submissions from the same participant across different accounts or teams will result in disqualification of all associated submissions.
Team mergers are permitted up to 5 days before the Phase 1 deadline and must be finalized through Kaggle's team-merger interface. No mergers are accepted after that.
At least one team member must be reachable at a working email address throughout the competition. Communication about advancement to Phase 2 will be sent to that address; teams unreachable within 24 hours of a Phase 2 invitation forfeit their slot.
2. Competition Structure — Two Phases
This is a two-phase competition. Read both phases before you start.

Phase 1 — Kaggle Leaderboard
Teams submit prediction CSVs directly to Kaggle. Public and private leaderboards are computed on separate splits of the test set given. Phase 1 closes on the date announced in the Timeline; the private leaderboard is revealed at that time.

Phase 2 — Solution Package Review and In-Person Final at IUT
After Phase 1 closes, the top 30 teams on the private leaderboard are invited to submit a complete solution package (specified in Section 3). Organizers run each package on a held-out fold of the dataset that is not part of the public or private Phase 1 test sets. From these results, the top 15 teams are invited to the in-person final at Islamic University of Technology (IUT), where final rankings are determined by a weighted combination of the following components:

Component	Weight
Phase-1 Private Leaderboard Score	20%
Phase-2 Held-out Test Set Score	50%
In-person Presentation	10%
Written Paper Report	10%
Novelty and Approach	10%
The final competition winners are decided by this combined score, not by the Kaggle private leaderboard alone. A team that leads the Kaggle private leaderboard but does not submit a working solution package by the Phase 2 deadline forfeits its place.

3. Submission Requirements
Phase 1 Submissions
Format: a CSV file with exactly two columns, id and label, matching the format of sample_submission.csv.
label must be 0 (hallucinated) or 1 (faithful) for every row. Missing rows, missing labels, non-integer values, or extra columns cause the submission to be rejected without a score.
Submissions are limited to 4 per day per team.
A team may select up to 2 submissions as their final scored submissions for the private leaderboard by the Phase 1 deadline. If no selection is made, the two most recent valid submissions are used.
Phase 2 Solution Package (top 30 teams only)
Teams invited to Phase 2 must submit, by the Phase 2 deadline:

A runnable Kaggle notebook that reproduces the team's Phase 1 predictions from the raw test set, end to end. The notebook must run within Kaggle's standard code-competition kernel limits: GPU (2*T4 or P100) allocation, and total runtime under 9 hours. Organizers will execute this notebook on the held-out fold; notebooks that do not run to completion in this environment are disqualified from Phase 2.
A 4-page paper report (excluding references) describing the approach, the methodology, and any analysis of results. Use ACL, EMNLP, or a similar standard NLP formatting template. The report must be a single PDF.
A brief README describing the environment, model weights used, and any external open-weight models loaded from Kaggle's model registry or Hugging Face mirrors.
Solution packages are submitted via a form linked from the Discussion tab at the start of Phase 2.

4. Compute and Inference Rules
Phase 2 scoring runs as a code competition. That means:

All Phase 2 code runs offline in Kaggle's kernel environment with no internet access.
Open-weight models only at inference. No calls to paid APIs, no proprietary model weights, no closed-source services. Models must be loadable from Kaggle datasets, the Kaggle Models registry, or Hugging Face mirrors that participants attach as datasets to their notebook.
Total inference runtime must be under 9 hours on a single P100 or 2*T4 GPU. Notebooks exceeding this limit are disqualified.
Total on-disk model size (all weights combined) must be under 50 GB.
For Phase 1, teams may generate predictions using any resources they have access to, provided the same predictions can be reproduced by a code-competition-compliant Phase 2 package. If your Phase 1 submissions rely on paid APIs and cannot be replicated offline, you will not be able to advance to Phase 2.

5. External Data and Models
Any publicly available Bengali or multilingual dataset may be used for training, validation, or feature engineering. Include a citation in your Phase 2 paper.

Any open-weight model — including Bengali-native models (TituLLM, TigerLLM), multilingual models (Llama, Qwen, Gemma, DeepSeek), and encoders (XLM-R, BanglaBERT) — may be used.

Teams may submit their own models. If your team has trained or fine-tuned its own model — whether a from-scratch Bengali model, a continued-pretraining variant, a task-specific fine-tune, or anything in between — you are welcome and encouraged to use it. The only requirement is that at Phase 2 the model must be loadable and runnable inside the offline Kaggle kernel within the compute limits in Section 4 (single P100 GPU or 2*T4 GPU, under 9 hours total runtime, under 50 GB of on-disk weights, no internet access at inference). Package your weights as a Kaggle dataset attached to your notebook. You are not required to publicly release the weights themselves — only to ensure the organizers can execute your notebook against the held-out fold.

The cached frontier-model outputs on the sample split, released with the competition data, are provided as a fair-play resource. All teams may use them freely. They are not released for the test split.

The Phase 2 held-out fold is drawn from a source distribution disclosed only after the competition ends. Teams may not attempt to reverse-engineer, reconstruct, or query this fold before it is released.

Data augmentation is allowed.

Fine-tuning on external Bengali data is allowed.

Fine-tuning on the labeled sample set is allowed.

Using the test set labels (which are not released) in any form is not allowed — obviously — but bears saying.

6. Fair-Play Conduct
No leaderboard probing beyond the intended purpose. Using submissions to reverse-engineer test-set labels (e.g., systematic label flipping across submissions to infer ground truth) is prohibited and results in disqualification.
No collusion between teams. Sharing code, predictions, or models privately between teams before the Phase 1 deadline is prohibited. Publicly sharing code, resources, or discussion via the Kaggle Discussion tab is encouraged.
No misrepresentation of authorship. Every team member listed on the final submission must have made a substantive contribution to the work. Adding names for the sake of prize distribution is prohibited.
Dataset issues must be reported publicly. If you believe a specific test item is mislabeled, has a contested ground truth, or contains a curation error, report it on the Discussion tab. Reporting privately to try to gain an advantage is prohibited; reporting publicly benefits everyone and is welcomed.
Organizers reserve the right to disqualify submissions that violate the spirit of these rules even where no specific clause is broken. This clause exists to catch things we could not anticipate.
7. Scoring, Ties, and Disputes
Primary metric: binary F1 on the HALLUCINATED class (label = 0). This is calculated over the private leaderboard rows only for Phase 1 final scoring, and over the held-out fold for the Phase 2 organizer run.
Tie-breaker (Phase 1): when two teams tie on binary F1, the team with the higher F1 on the C1 cultural-distance subset wins the tie. If still tied, the team whose earliest qualifying submission came first breaks the tie.
Score disputes must be raised via the Discussion tab or by email to the organizing team within 72 hours of the private leaderboard reveal. Disputes raised later are not considered.
Organizers will publish a post-competition analysis notebook showing per-band and per-domain performance across all top-30 teams. This is analytical, not adjudicatory — final rankings are set by the weighted scheme in Section 2.
8. Eligibility
Participant must be a currently enrolled undergraduate student from any recognized university in Bangladesh. Employees of the organizing institutions (Institute of Policy Dynamics and IUTCS) and their immediate families are not eligible for prize money but may participate in the leaderboard and be credited in the paper. Sponsor employees are eligible.
Participants must be at least 13 years of age, per Kaggle's terms of service. Participants under the age of 18 need parental or guardian consent for prize acceptance and travel to the in-person final.
Travel to the in-person final at IUT is expected of top-15 teams. A remote participation option will be provided for teams unable to travel; remote teams present via video conference and receive the presentation and paper components of the score identically. Prize money and travel support policies are announced in the Discussion tab at Phase 2 invitations.
9. Timeline
Specific dates are in the Timeline tab. All deadlines are in Bangladesh Standard Time (UTC+6). Deadlines are strict; extensions are announced publicly on the Discussion tab if granted for infrastructure reasons and apply equally to all teams.

10. Changes to These Rules
Organizers may amend these rules to correct ambiguity, address unanticipated situations, or improve fairness. Any material change will be announced on the Discussion tab with at least 1 day's notice before it takes effect, unless the change is a bug fix or clarification that no reasonable team could argue disadvantages them.
