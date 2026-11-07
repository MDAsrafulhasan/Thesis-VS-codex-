from __future__ import annotations

import re

from datathon_utils import clean_context, normalize_bn
from fuzzy_source_overrides import relation_signature, token_set
from source_answer_bank import canonical_answer


CLAUSE_RE = re.compile(r"[।!?;\n]+")
NUMERIC_QUESTION_RE = re.compile(
    r"(?:"
    r"কত\s*(?:সালে|সাল|তারিখ|গুলি|গুলো|টি|জন|ফুট|কিলোমিটার|রকম)"
    r"|কবে"
    r"|কোন\s+(?:সালে|বছর)"
    r"|জন্মসাল\s+কত"
    r"|সংখ্যা\s+কত"
    r"|দূরত্ব\s+কত"
    r"|আয়তন.*কত"
    r")"
)
STOPWORDS = frozenset(
    normalize_bn(
        "কী কি কে কার কাকে কোথায় কোথায় কবে কত কোন কোনটি সালে সাল "
        "তার তিনি এটি ছিল হয় হয় হয়েছিল হয়েছে করেন করে করা থেকে জন্য "
        "এবং এর একটি বর্তমানে মোট নাম বলো উত্তর"
    ).split()
)


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in token_set(text)
        if len(token) >= 3 and token not in STOPWORDS
    }


def context_span_decision(
    prompt: str,
    response: str,
    context: str,
    *,
    min_shared_anchors: int = 2,
) -> tuple[int | None, dict]:
    """Confirm an exact answer span only inside a relation-compatible clause."""
    context = clean_context(context)
    answer = canonical_answer(response)
    normalized_prompt = normalize_bn(prompt)
    if not context or len(answer) < 3:
        return None, {}
    # Numeric/date binding needs a dedicated parser: an exact year can occur in
    # the correct subject's clause while describing birth, death, acquisition,
    # publication, or establishment. That family is handled earlier by the
    # relation-aware numeric and date stages.
    if NUMERIC_QUESTION_RE.search(normalized_prompt):
        return None, {}

    prompt_anchors = content_tokens(prompt) - token_set(answer)
    prompt_relations = relation_signature(prompt)
    candidates = []
    for raw_clause in CLAUSE_RE.split(context):
        clause = normalize_bn(raw_clause)
        if not clause or answer not in canonical_answer(clause):
            continue
        shared = prompt_anchors & token_set(clause)
        clause_relations = relation_signature(clause)
        if prompt_relations and not prompt_relations.issubset(clause_relations):
            continue
        candidates.append(
            {
                "clause": raw_clause.strip(),
                "shared_anchors": sorted(shared),
                "shared_anchor_count": len(shared),
                "prompt_relations": sorted(prompt_relations),
                "clause_relations": sorted(clause_relations),
            }
        )
    if not candidates:
        return None, {}
    best = max(candidates, key=lambda item: item["shared_anchor_count"])
    if best["shared_anchor_count"] < min_shared_anchors:
        return None, {}
    return 1, {
        "rule": "relation_guarded_exact_context_span",
        "answer": response,
        "min_shared_anchors": min_shared_anchors,
        **best,
    }
