# Phase 2 organizer-run runtime evidence

This note records the completed Phase 2 inference run contained in
`huntrix-1.ipynb`, supplied to the team after organizer execution on July 22,
2026. It reports runtime and routing behavior. The organizers later released
an aggregate Phase 2 score, but no row-level ground-truth labels.

## Execution identity

| Field | Value |
|---|---|
| Notebook | `huntrix-1.ipynb` |
| Notebook SHA-256 | `41096e5d9b2f293ff6b42bb04572d8b25f6353f75e1b3b5a27bc13d775ab4d91` |
| Input rows | 5,000 |
| GPUs | Tesla T4 x2 |
| Model | Qwen3.6-27B Q4_K_P |
| Model size | 16.332 GiB |
| Output contract | 5,000 ordered IDs; non-null labels in `{0,1}` |
| Completion status | Successful |

## Measured runtime

| Measurement | Seconds | Wall-clock form |
|---|---:|---:|
| Router complete | 73.1774 | 1 min 13.18 sec |
| Model ready, cumulative from start | 171.2503 | 2 min 51.25 sec |
| Model startup after routing | 98.0729 | 1 min 38.07 sec |
| Judge stage | 13,747.5044 | 3 h 49 min 7.50 sec |
| End-to-end total | 13,921.9134 | 3 h 52 min 1.91 sec |

The run used 42.97% of the nine-hour limit and retained 18,478.09 seconds, or
5 hours 7 minutes 58.09 seconds, of headroom.

## Routing distribution

| Route outcome | Rows | Share |
|---|---:|---:|
| Deterministically resolved | 207 | 4.14% |
| Sent to the Qwen judge | 4,793 | 95.86% |

The deterministic stage counts were:

| Stage | Rows |
|---|---:|
| Context rule | 166 |
| Exact answer bank | 21 |
| Fuzzy answer bank | 12 |
| Weighted answer bank | 5 |
| Typed idiom | 2 |
| Context span | 1 |

The judge changed 2,325 of the router's provisional labels. The final output
contained 2,582 zeros and 2,418 ones. These counts describe system behavior;
they are not evidence of accuracy or class prevalence in the hidden labels.

## Interpretation

Phase 1 deterministic coverage was 93.72%, but Phase 2 coverage fell to 4.14%.
This confirms that the held-out source distribution differed substantially from
the Phase 1 source lineage. The router responded conservatively by abstaining
instead of forcing approximate matches. The fixed Qwen fallback absorbed the
larger queue and completed within the required runtime.

The official Phase 2 standings released on July 23 report a score of `0.90144`
for Huntrix, ranking the team seventh among 32 entries. This establishes the
aggregate outcome, but it does not identify which rows, domains, or routing
branches produced the remaining errors.
