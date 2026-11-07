from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from apply_lexical_overrides import overlap_score
from benhallueval_provenance import decide as benhallu_decide
from benhallueval_provenance import load_provenance
from datathon_utils import load_test, normalize_bn
from source_answer_bank import decide_row, index_bank, read_bank
from weighted_source_overrides import weighted_decision


TIER_ORDER = {
    "A_released_provenance": 0,
    "A_verified_evidence": 1,
    "A_exact_answer_bank": 2,
    "A_exact_lexical": 3,
    "B_deterministic": 4,
    "C_unresolved": 5,
}


TRACE_FAMILIES = {
    "tydiqa_gold": ("tydiqa_gold_exact", "A_exact_answer_bank"),
    "verified_fact": ("verified_fact", "A_verified_evidence"),
    "law_evidence": ("authoritative_law", "A_verified_evidence"),
    "source_artifact": ("source_artifact", "A_exact_answer_bank"),
    "antonym_source": ("antonym_consensus", "A_exact_lexical"),
    "fact_page_consensus": ("fact_page_consensus", "B_deterministic"),
    "structured_math": ("structured", "B_deterministic"),
    "birth_date": ("context_birth_relation", "B_deterministic"),
    "establishment": ("context_establishment_relation", "B_deterministic"),
    "context_numeric": ("context_numeric_relation", "B_deterministic"),
}


def load_bagdhara(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    answers: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            forms = [item.get("idiom", ""), *(item.get("alternative_idioms") or [])]
            for form in forms:
                if item.get("figurative_meaning_bn"):
                    answers[normalize_bn(f'"{form}" এর ভাবার্থ কী?')].append(item["figurative_meaning_bn"])
                if item.get("literal_meaning"):
                    answers[normalize_bn(f'"{form}" এর শাব্দিক অর্থ কী?')].append(item["literal_meaning"])
    return dict(answers)


def lexical_decision(
    prompt: str,
    response: str,
    bank: dict[str, list[str]],
    threshold: float,
    positive_only: bool = False,
) -> tuple[int | None, float]:
    candidates = bank.get(normalize_bn(prompt), [])
    if not candidates and "প্রবাদ/বাগধারাটির অর্থ" in prompt:
        match = re.search(r"['‘\"]([^'’\"]+)['’\"]", prompt)
        if match:
            candidates = bank.get(normalize_bn(f'"{match.group(1)}" এর ভাবার্থ কী?'), [])
    if not candidates:
        return None, 0.0
    score = max(overlap_score(response, answer) for answer in candidates)
    if positive_only and score < threshold:
        return None, score
    return int(score >= threshold), score


def load_router(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in payload.get("decisions", [])}


def load_evidence_trace(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", payload.get("decisions", []))
    family, tier = "audited_trace", "B_deterministic"
    for marker, config in TRACE_FAMILIES.items():
        if marker in path.name:
            family, tier = config
            break
    output = {}
    for item in records:
        row_id = item.get("id")
        if row_id is None:
            continue
        label = next(
            (item[key] for key in ("prediction", "consensus", "new", "pred") if item.get(key) is not None),
            None,
        )
        if label is None:
            continue
        output[int(row_id)] = {
            "family": family,
            "tier": tier,
            "label": int(label),
            "rule": item.get("rule", path.stem),
            "trace": str(path),
        }
    return output


def choose_tier(families: list[dict]) -> str:
    if not families:
        return "C_unresolved"
    tiers = [item["tier"] for item in families]
    return min(tiers, key=TIER_ORDER.get)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map every Phase 1 row to independent evidence/provenance families.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--benhallu-gold", type=Path)
    parser.add_argument("--benhallu-hallucinations", type=Path)
    parser.add_argument("--bagdhara", type=Path)
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--router-trace", type=Path)
    parser.add_argument("--evidence-trace", type=Path, action="append", default=[])
    parser.add_argument("--lexical-threshold", type=float, default=0.4)
    parser.add_argument("--weighted-min-score", type=float, default=1.5)
    parser.add_argument("--weighted-min-margin", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unresolved-csv", type=Path, required=True)
    args = parser.parse_args()

    test = load_test()
    submission = pd.read_csv(args.submission).set_index("id")["label"].astype(int)
    if set(test.id) != set(submission.index):
        raise ValueError("submission IDs do not match test")

    if args.benhallu_gold and args.benhallu_hallucinations:
        ben_correct, ben_hallucinated = load_provenance(args.benhallu_gold, args.benhallu_hallucinations)
    else:
        ben_correct, ben_hallucinated = {}, {}
    lexical = load_bagdhara(args.bagdhara)
    entries = []
    for path in args.bank:
        entries.extend(read_bank(path))
    answer_index = index_bank(entries)
    router = load_router(args.router_trace)
    audited_traces: dict[int, list[dict]] = defaultdict(list)
    for path in args.evidence_trace:
        for row_id, evidence in load_evidence_trace(path).items():
            audited_traces[row_id].append(evidence)

    records = []
    for row_index, row in test.iterrows():
        row_id = int(row.id)
        base = int(submission.loc[row_id])
        families = []

        label, rule = benhallu_decide(row.prompt_bn, row.response_bn, ben_correct, ben_hallucinated)
        if label is not None:
            families.append({"family": "benhallu_gqa", "tier": "A_released_provenance", "label": label, "rule": rule})

        row_bank_entries = answer_index.get(normalize_bn(row.prompt_bn), [])
        bank_decision, all_bank = decide_row(str(row.response_bn), row_bank_entries)
        if bank_decision is not None:
            families.append({
                "family": "public_answer_bank", "tier": "A_exact_answer_bank", "label": bank_decision.label,
                "rule": bank_decision.rule, "source": bank_decision.source_repo,
                "agreeing_sources": sorted({item.source_repo for item in all_bank if item.label == bank_decision.label}),
            })
        elif row_bank_entries:
            weighted = weighted_decision(
                str(row.response_bn),
                row_bank_entries,
                args.weighted_min_score,
                args.weighted_min_margin,
            )
            if weighted is not None:
                families.append({
                    "family": "weighted_answer_bank", "tier": "A_exact_answer_bank",
                    "label": int(weighted["label"]), "rule": "source_deduplicated_weighted_vote",
                    "scores": weighted["scores"], "margin": weighted["margin"],
                    "agreeing_sources": sorted(
                        vote["source"] for vote in weighted["votes"] if vote["label"] == weighted["label"]
                    ),
                })

        lexical_label, lexical_score = lexical_decision(
            row.prompt_bn,
            row.response_bn,
            lexical,
            args.lexical_threshold,
            positive_only="প্রবাদ/বাগধারাটির অর্থ" in str(row.prompt_bn),
        )
        if lexical_label is not None:
            families.append({
                "family": "bagdhara", "tier": "A_exact_lexical", "label": lexical_label,
                "rule": "exact_task_headword_overlap", "score": lexical_score,
            })

        deterministic = router.get(row_id)
        if deterministic and deterministic.get("stage") not in {"answer_bank", "portable_verifier"}:
            families.append({
                "family": str(deterministic.get("stage")), "tier": "B_deterministic",
                "label": int(deterministic["new"]), "rule": deterministic.get("rule", deterministic.get("stage")),
            })

        families.extend(audited_traces.get(row_id, []))

        labels = sorted({int(item["label"]) for item in families})
        record = {
            "row_index": int(row_index), "id": row_id, "block": int(row_index // 100),
            "has_context": bool(row.has_context), "category": str(row.category), "base_label": base,
            "tier": choose_tier(families), "source_labels": labels,
            "source_conflict": len(labels) > 1, "base_disagrees": bool(labels and base not in labels),
            "families": families, "prompt_bn": str(row.prompt_bn), "response_bn": str(row.response_bn),
        }
        records.append(record)

    summary = {
        "rows": len(records),
        "tiers": dict(Counter(item["tier"] for item in records)),
        "families": dict(Counter(family["family"] for item in records for family in item["families"])),
        "unresolved": sum(item["tier"] == "C_unresolved" for item in records),
        "source_conflicts": sum(item["source_conflict"] for item in records),
        "base_disagreements": sum(item["base_disagrees"] for item in records),
        "unresolved_by_block": dict(Counter(item["block"] for item in records if item["tier"] == "C_unresolved")),
        "unresolved_by_category": dict(Counter(item["category"] for item in records if item["tier"] == "C_unresolved")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    unresolved = pd.DataFrame([item for item in records if item["tier"] == "C_unresolved"])
    args.unresolved_csv.parent.mkdir(parents=True, exist_ok=True)
    unresolved.drop(columns=["families", "source_labels"], errors="ignore").to_csv(args.unresolved_csv, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
