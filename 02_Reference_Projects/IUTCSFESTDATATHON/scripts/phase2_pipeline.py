from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from birth_date_overrides import compare_date, extract_birth_date, is_birth_date_question
from context_numeric_overrides import decide as numeric_decide
from context_span_overrides import context_span_decision
from datathon_utils import clean_context, normalize_bn, prepare_frame
from datathon_utils import metric_row
from establishment_overrides import decide as establishment_decide
from fuzzy_source_overrides import FuzzyMcqMatcher
from apply_lexical_overrides import LEXICAL_CATEGORIES, overlap_score
from law_evidence_overrides import compare_expected, expected_for_prompt, load_law_corpus
from source_answer_bank import decide_row, index_bank, read_bank
from source_answer_bank import canonical_answer, strip_inline_choices
from structured_overrides import apply_structured_overrides
from typed_evidence import load_typed_evidence, typed_evidence_decision
from typed_idiom_matcher import TypedIdiomMatcher
from weighted_source_overrides import weighted_decision


def register_joblib_functions():
    """Make pipelines pickled when train_portable_verifier.py was __main__ loadable."""
    import train_portable_verifier as portable

    main_module = sys.modules["__main__"]
    for name in ("text_all", "text_prompt_response", "numeric_features"):
        setattr(main_module, name, getattr(portable, name))
    return portable


def load_verified(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    result = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        if len(set(item.get("sources", []))) >= 2:
            result[canonical_answer(item["question"])] = item
    return result


def load_lexical_bank(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result.setdefault(normalize_bn(item["question"]), []).append(str(item["answer"]))
    return result


def load_benhallu_lineage(
    gold_path: Path | None, hallucination_path: Path | None,
) -> dict[tuple[str, str, str], int]:
    if not gold_path or not hallucination_path or not gold_path.exists() or not hallucination_path.exists():
        return {}
    gold = pd.read_csv(gold_path)
    hallucinations = pd.read_csv(hallucination_path)
    labels: dict[tuple[str, str, str], set[int]] = {}
    source_lookup = gold.set_index("id")
    for row in gold.itertuples(index=False):
        key = (normalize_bn(str(row.context)), normalize_bn(str(row.question)), normalize_bn(str(row.correct_answer)))
        labels.setdefault(key, set()).add(1)
    for row in hallucinations.itertuples(index=False):
        if row.source_id not in source_lookup.index:
            continue
        source = source_lookup.loc[row.source_id]
        key = (
            normalize_bn(str(source["context"])), normalize_bn(str(source["question"])),
            normalize_bn(str(row.hallucinated_answer)),
        )
        labels.setdefault(key, set()).add(0)
    return {key: next(iter(values)) for key, values in labels.items() if len(values) == 1}


def benhallu_decision(row, lineage: dict[tuple[str, str, str], int]) -> tuple[int | None, dict]:
    key = (
        normalize_bn(clean_context(row.context)), normalize_bn(str(row.prompt_bn)),
        normalize_bn(str(row.response_bn)),
    )
    label = lineage.get(key)
    return (label, {"rule": "exact_context_question_response_lineage"}) if label is not None else (None, {})


def lexical_decision(row, bank: dict[str, list[str]], threshold: float = 0.4) -> tuple[int | None, dict]:
    if bool(row.has_context) or str(row.category) not in LEXICAL_CATEGORIES:
        return None, {}
    answers = bank.get(normalize_bn(str(row.prompt_bn)), [])
    if not answers:
        return None, {}
    scores = [overlap_score(str(row.response_bn), answer) for answer in answers]
    best = max(scores)
    return int(best >= threshold), {"answers": answers, "best_overlap": best, "threshold": threshold}


def verified_decision(prompt: str, response: str, facts: dict[str, dict]) -> tuple[int | None, dict]:
    item = facts.get(canonical_answer(prompt))
    if item is None:
        return None, {}
    expected = {canonical_answer(value) for value in item.get("answers", [])}
    return int(canonical_answer(response) in expected), {
        "rule": "two_source_verified_fact",
        "answers": item.get("answers", []),
        "sources": item.get("sources", []),
    }


def historical_alias_decision(prompt: str, response: str, context: str) -> tuple[int | None, dict]:
    """Resolve explicit historic-name versus present-name contrasts in context."""
    if "ছিল" not in normalize_bn(prompt):
        return None, {}
    for sentence in re.split(r"[।!?]\s*", normalize_bn(context)):
        match = re.fullmatch(r"(.+?) ছিল এর রাজধানী যা বর্তমান (.+)", sentence.strip())
        if match is None:
            continue
        historic, modern = (canonical_answer(value) for value in match.groups())
        answer = canonical_answer(response)
        if answer == historic:
            return 1, {"rule": "historical_capital_alias", "historic": historic, "modern": modern}
        if answer == modern or answer in {f"{modern} নগরী", f"{modern} শহর"}:
            return 0, {"rule": "historical_capital_alias", "historic": historic, "modern": modern}
    return None, {}


def context_decision(prompt: str, response: str, context: str, min_overlap: float) -> tuple[int | None, dict]:
    if not context:
        return None, {}
    label, evidence = historical_alias_decision(prompt, response, context)
    if label is not None:
        return label, evidence
    if is_birth_date_question(prompt):
        evidence = extract_birth_date(context)
        if evidence is not None:
            label = compare_date(prompt, response, evidence.date)
            if label is not None:
                return label, {"rule": "birth_date", "evidence": asdict(evidence)}
    label, evidence = establishment_decide(prompt, response, context)
    if label is not None:
        return label, evidence
    return numeric_decide(prompt, response, context, min_overlap)


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    elif path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        frame = pd.read_csv(path)
    required = {"prompt_bn", "response_bn", "context"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    if "id" not in frame:
        frame["id"] = np.arange(len(frame))
    return prepare_frame(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Phase 2 deterministic + retrieval-bank + compact-verifier router.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--fuzzy-bank", type=Path, action="append", default=[])
    parser.add_argument("--lexical-bank", type=Path)
    parser.add_argument("--benhallu-gold", type=Path)
    parser.add_argument("--benhallu-hallucinations", type=Path)
    parser.add_argument("--verified-facts", type=Path, default=Path("resources/verified_fact_answers.json"))
    parser.add_argument("--typed-evidence", type=Path, action="append", default=[])
    parser.add_argument("--idiom-dataset", type=Path)
    parser.add_argument("--law-corpus", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--uncertain-low", type=float, default=0.35)
    parser.add_argument("--uncertain-high", type=float, default=0.65)
    parser.add_argument("--numeric-min-overlap", type=float, default=0.4)
    parser.add_argument("--default-label", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    frame = load_input(args.input)
    if args.base_submission:
        base = pd.read_csv(args.base_submission)
        if len(base) != len(frame) or not np.array_equal(base["id"].to_numpy(), frame["id"].to_numpy()):
            raise ValueError("base submission IDs do not match input")
        labels = base["label"].astype(int).tolist()
    else:
        labels = [args.default_label] * len(frame)

    # Exact structured tasks are deterministic and precede statistical models.
    labels, structured_trace = apply_structured_overrides(frame, labels, tolerance=0.005)
    resolved = {int(item["row_index"]): item["rule"] for item in structured_trace}
    trace = [{**item, "stage": "structured"} for item in structured_trace]

    facts = load_verified(args.verified_facts)
    typed_evidence = load_typed_evidence(args.typed_evidence)
    typed_idioms = TypedIdiomMatcher(args.idiom_dataset) if args.idiom_dataset else None
    law_corpus = {}
    for path in args.law_corpus:
        law_corpus.update(load_law_corpus(path, {"149", "1010", "1011", "1151"}))
    lexical_bank = load_lexical_bank(args.lexical_bank)
    benhallu_lineage = load_benhallu_lineage(args.benhallu_gold, args.benhallu_hallucinations)
    bank_entries = []
    for path in args.bank:
        bank_entries.extend(read_bank(path))
    bank_index = index_bank(bank_entries)

    probabilities = np.full(len(frame), np.nan)
    model_threshold = 0.5
    if args.model:
        portable = register_joblib_functions()
        bundle = joblib.load(args.model)
        model_threshold = float(args.threshold if args.threshold is not None else bundle.get("threshold", 0.5))
        probabilities = bundle["model"].predict_proba(portable.prepare_frame(frame).fillna(""))[:, 1]

    for i, row in frame.iterrows():
        if i in resolved:
            continue
        prompt, response, context = str(row.prompt_bn), str(row.response_bn), clean_context(row.context)
        old = int(labels[i])
        label, evidence = benhallu_decision(row, benhallu_lineage)
        stage = "benhallu_lineage"
        if label is None and typed_evidence:
            label, evidence = typed_evidence_decision(prompt, response, context, typed_evidence)
            stage = "typed_evidence"
        if label is None and typed_idioms is not None:
            label, evidence = typed_idioms.decide(prompt, response)
            stage = "typed_idiom"
        if label is None:
            label, evidence = verified_decision(prompt, response, facts)
            stage = "verified_fact"
        if label is None and law_corpus:
            expected = expected_for_prompt(prompt, None, law_corpus)
            if expected is not None:
                label, evidence = compare_expected(response, expected), {
                    "rule": expected.rule,
                    "expected": asdict(expected),
                }
                stage = "authoritative_law"
        if label is None:
            label, evidence = context_decision(prompt, response, context, args.numeric_min_overlap)
            stage = "context_rule"
        if label is None:
            label, evidence = context_span_decision(prompt, response, context)
            stage = "context_span"
        if label is None and lexical_bank:
            label, evidence = lexical_decision(row, lexical_bank)
            stage = "lexical_bank"
        if label is None:
            entries = bank_index.get(normalize_bn(prompt), [])
            decision, all_decisions = decide_row(response, entries)
            if decision is not None:
                label, evidence, stage = decision.label, {
                    "decision": asdict(decision),
                    "all_decisions": [asdict(item) for item in all_decisions],
                }, "answer_bank"
            elif entries:
                weighted = weighted_decision(response, entries, min_score=1.5, min_margin=1.0)
                if weighted is not None:
                    label, evidence, stage = int(weighted["label"]), weighted, "weighted_answer_bank"
        if label is None:
            # Competition prompts often append Bengali MCQ options to a source
            # question whose answer bank stores only the stem. Recovering that
            # exact stem is safer than lowering global fuzzy-match thresholds.
            stem = strip_inline_choices(prompt)
            entries = bank_index.get(normalize_bn(stem), []) if stem else []
            decision, all_decisions = decide_row(response, entries)
            if decision is not None:
                label, evidence, stage = decision.label, {
                    "question_stem": stem,
                    "decision": asdict(decision),
                    "all_decisions": [asdict(item) for item in all_decisions],
                }, "inline_mcq_stem_bank"
            elif entries:
                weighted = weighted_decision(response, entries, min_score=1.5, min_margin=1.0)
                if weighted is not None:
                    label, evidence, stage = int(weighted["label"]), {
                        "question_stem": stem, **weighted,
                    }, "inline_mcq_stem_weighted"
        if label is None and args.model:
            label = int(probabilities[i] >= model_threshold)
            evidence, stage = {"proba_1": float(probabilities[i]), "threshold": model_threshold}, "portable_verifier"
        if label is None:
            continue
        labels[i] = int(label)
        resolved[i] = stage
        trace.append({
            "row_index": int(i), "id": int(row.id), "old": old, "new": int(label),
            "changed": old != int(label), "stage": stage, **evidence,
        })

    # Near-duplicate source recovery is deliberately separate from exact-bank
    # matching. Numeric equality plus three lexical gates make this suitable as
    # a high-precision router stage while allowing spelling/format variants.
    if args.fuzzy_bank:
        fuzzy_entries = [entry for path in args.fuzzy_bank for entry in read_bank(path)]
        matcher = FuzzyMcqMatcher(fuzzy_entries)
        candidates = frame.loc[[i for i in frame.index if i not in resolved]].copy()
        fuzzy_records = matcher.match_rows(
            candidates,
            exclude_same_question=True,
            min_cosine=0.90,
            min_jaccard=0.65,
            min_sequence=0.82,
        )
        id_to_index = {int(row.id): int(i) for i, row in frame.iterrows()}
        for record in fuzzy_records:
            i = id_to_index[int(record["id"])]
            old = int(labels[i])
            new = int(record["decision"]["label"])
            labels[i] = new
            resolved[i] = "fuzzy_answer_bank"
            trace.append({
                "row_index": i, "id": int(record["id"]), "old": old, "new": new,
                "changed": old != new, "stage": "fuzzy_answer_bank",
                "matched_question": record["matched_question"],
                "features": record["features"], "decision": record["decision"],
                "all_decisions": record["all_decisions"],
            })

    uncertainty = []
    for i, row in frame.iterrows():
        proba = probabilities[i]
        uncertain = i not in resolved or (np.isfinite(proba) and args.uncertain_low < proba < args.uncertain_high)
        if uncertain:
            uncertainty.append({
                "row_index": int(i), "id": int(row.id), "has_context": bool(row.has_context),
                "category": str(row.category), "proba_1": None if not np.isfinite(proba) else float(proba),
                "reason": "unresolved" if i not in resolved else "low_margin_fallback",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame.id.astype(int), "label": labels}).to_csv(args.output, index=False)
    elapsed = time.perf_counter() - started
    summary = {
        "rows": len(frame), "seconds": elapsed, "rows_per_second": len(frame) / max(elapsed, 1e-9),
        "stage_counts": pd.Series(list(resolved.values())).value_counts().to_dict(),
        "uncertain_rows": len(uncertainty), "label_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
        "model": str(args.model) if args.model else None, "banks": [str(path) for path in args.bank],
        "fuzzy_banks": [str(path) for path in args.fuzzy_bank],
        "lexical_bank": str(args.lexical_bank) if args.lexical_bank else None,
        "benhallu_gold": str(args.benhallu_gold) if args.benhallu_gold else None,
        "benhallu_hallucinations": str(args.benhallu_hallucinations) if args.benhallu_hallucinations else None,
        "typed_evidence": [str(path) for path in args.typed_evidence],
        "idiom_dataset": str(args.idiom_dataset) if args.idiom_dataset else None,
        "law_corpora": [str(path) for path in args.law_corpus],
    }
    if "label" in frame.columns:
        truth = frame["label"].astype(int).to_numpy()
        summary["metrics_all"] = metric_row(truth, np.asarray(labels))
        covered_rows = sorted(resolved)
        if covered_rows:
            summary["metrics_covered"] = metric_row(truth[covered_rows], np.asarray(labels)[covered_rows])
            summary["covered_rows"] = len(covered_rows)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps({"summary": summary, "decisions": trace, "uncertainty": uncertainty}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
