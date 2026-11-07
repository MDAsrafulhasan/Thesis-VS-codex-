from __future__ import annotations

import json
import re
from pathlib import Path

from apply_lexical_overrides import overlap_score
from source_answer_bank import canonical_answer
from typed_evidence import infer_intent


QUOTED_IDIOM_RE = re.compile(r"[\"'“‘]([^\"'”’]+)[\"'”’]")
DASH_IDIOM_RE = re.compile(r"^\s*(.+?)\s*[–—-]\s*বাগধার")


def extract_typed_idiom(prompt: str) -> str | None:
    if infer_intent(prompt) not in {"figurative_meaning", "literal_meaning"}:
        return None
    quoted = QUOTED_IDIOM_RE.search(str(prompt))
    if quoted:
        return canonical_answer(quoted.group(1))
    dashed = DASH_IDIOM_RE.search(str(prompt))
    if dashed:
        return canonical_answer(dashed.group(1))
    return None


def load_idioms(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        alternatives = item.get("alternative_idioms") or []
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        for match_kind, value in [
            ("primary", item.get("idiom", "")),
            *(("alias", value) for value in alternatives),
        ]:
            key = canonical_answer(value)
            if key:
                result.setdefault(key, []).append(
                    {"item": item, "match_kind": match_kind}
                )
    return result


class TypedIdiomMatcher:
    """High-precision literal/figurative matcher over the expert idiom corpus."""

    def __init__(
        self,
        dataset: Path,
        *,
        min_requested_overlap: float = 0.60,
        min_opposite_overlap: float = 0.50,
        min_margin: float = 0.15,
    ):
        self.dataset = dataset
        self.idioms = load_idioms(dataset)
        self.min_requested_overlap = min_requested_overlap
        self.min_opposite_overlap = min_opposite_overlap
        self.min_margin = min_margin

    def decide(self, prompt: str, response: str) -> tuple[int | None, dict]:
        idiom = extract_typed_idiom(prompt)
        intent = infer_intent(prompt)
        matches = self.idioms.get(idiom or "")
        if not matches or intent not in {"figurative_meaning", "literal_meaning"}:
            return None, {}

        # A headword's own record is stronger evidence than another idiom naming
        # it as an alternative. Keep every duplicate primary record so conflicting
        # dictionary senses cause abstention instead of depending on file order.
        primary_matches = [match for match in matches if match["match_kind"] == "primary"]
        selected_matches = primary_matches or matches
        decisions = [
            self._decide_match(match, intent=intent, response=response)
            for match in selected_matches
        ]
        if any(decision[0] is None for decision in decisions):
            return None, {}
        labels = {decision[0] for decision in decisions}
        if len(labels) != 1:
            return None, {}

        label = labels.pop()
        evidence_items = [decision[1] for decision in decisions]
        return label, {
            "rule": evidence_items[0]["rule"],
            "intent": intent,
            "idiom": idiom,
            "match_kind": selected_matches[0]["match_kind"],
            "matching_records": len(selected_matches),
            "records": evidence_items,
            "thresholds": {
                "min_requested_overlap": self.min_requested_overlap,
                "min_opposite_overlap": self.min_opposite_overlap,
                "min_margin": self.min_margin,
            },
            "source_dataset": str(self.dataset),
        }

    def _decide_match(
        self,
        match: dict,
        *,
        intent: str,
        response: str,
    ) -> tuple[int | None, dict]:
        item = match["item"]
        figurative = str(item.get("figurative_meaning_bn", ""))
        literal = str(item.get("literal_meaning", ""))
        figurative_overlap = overlap_score(response, figurative)
        literal_overlap = overlap_score(response, literal)
        if intent == "figurative_meaning":
            requested_overlap, opposite_overlap = figurative_overlap, literal_overlap
            requested_field, opposite_field = "figurative_meaning_bn", "literal_meaning"
        else:
            requested_overlap, opposite_overlap = literal_overlap, figurative_overlap
            requested_field, opposite_field = "literal_meaning", "figurative_meaning_bn"

        label: int | None = None
        rule = ""
        if (
            requested_overlap >= self.min_requested_overlap
            and requested_overlap - opposite_overlap >= self.min_margin
        ):
            label = 1
            rule = "typed_idiom_requested_sense"
        elif (
            opposite_overlap >= self.min_opposite_overlap
            and opposite_overlap - requested_overlap >= self.min_margin
        ):
            label = 0
            rule = "typed_idiom_opposite_sense"
        if label is None:
            return None, {}
        return label, {
            "rule": rule,
            "source_idiom": item.get("idiom"),
            "requested_field": requested_field,
            "opposite_field": opposite_field,
            "figurative_meaning": figurative,
            "literal_meaning": literal,
            "figurative_overlap": figurative_overlap,
            "literal_overlap": literal_overlap,
        }
