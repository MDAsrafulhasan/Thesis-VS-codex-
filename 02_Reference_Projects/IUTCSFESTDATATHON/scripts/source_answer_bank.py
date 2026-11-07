from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shutil
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import hf_hub_download

from datathon_utils import OUTPUT_DIR, load_sample, load_test, metric_row, normalize_bn


HISHAB_REPO = "hishab/bangla-mmlu"
HISHAB_FILES = [
    "data/dev-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet",
]
SOMAJ_REPO = "farihashifa/SOMAJGYAAN"
SOMAJ_FILE = "bqad2025.csv"
BLUCK_ZIP = "https://github.com/minhaj1403/bluck/archive/refs/heads/main.zip"

PUNCT_RE = re.compile(r"[^0-9a-z\u0980-\u09ff@%+\-√π°<>=.]+")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
SCALE_WORDS = {"কোটি", "লক্ষ", "লাখ", "হাজার", "মিলিয়ন", "মিলিয়ন", "বিলিয়ন", "বিলিয়ন"}
YEAR_SUFFIX_RE = re.compile(
    r"(?:খ্রিষ্টাব্দে?|খ্রিস্টাব্দে?|খৃ|খ্রি|সালে?|সনের|সন)(?=$|\s)", flags=re.I
)
COUNT_SUFFIX_RE = re.compile(r"(?<=\d)\s*(?:টি|জন|খানা|তম)(?=$|\s)")
DAY_ORDINAL_SUFFIX_RE = re.compile(r"(?<=\d)(?:লা|রা|ঠা|ই|শে)(?=$|\s|[,।])")
INLINE_CHOICE_RE = re.compile(r"\s+[কখগঘ][\)\].:]")


@dataclass(frozen=True)
class BankEntry:
    question: str
    answers: list[str]
    choices: list[str]
    correct_index: int | None
    source_repo: str
    source_file: str
    license: str
    source_kind: str


@dataclass(frozen=True)
class Decision:
    label: int
    confidence: float
    rule: str
    source_repo: str
    answer: str


def canonical_answer(text: object) -> str:
    value = html.unescape(unicodedata.normalize("NFKC", str(text or "")))
    value = value.translate(ASCII_DIGITS).lower()
    # Preserve mathematical signs. Dropping a minus can collapse distinct MCQ
    # options such as ``x < -1`` and ``x < 1`` into the same canonical answer.
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("&gt", ">").replace("&lt", "<")
    value = re.sub(r"[\u200b-\u200d\ufeff]", "", value)
    value = YEAR_SUFFIX_RE.sub(" ", value)
    value = COUNT_SUFFIX_RE.sub("", value)
    value = DAY_ORDINAL_SUFFIX_RE.sub("", value)
    value = PUNCT_RE.sub(" ", value)
    value = re.sub(r"\s*([.+<>=])\s*", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value


def strip_inline_choices(question: object) -> str | None:
    """Return the question stem when Bengali MCQ choices are embedded inline."""
    value = str(question or "")
    match = INLINE_CHOICE_RE.search(value)
    if match is None:
        return None
    stem = value[: match.start()].strip()
    return stem or None


def structural_answer(text: object) -> str:
    """Canonicalize an MCQ option without erasing mathematical structure."""
    value = html.unescape(unicodedata.normalize("NFKC", str(text or "")))
    value = value.translate(ASCII_DIGITS).lower()
    value = value.replace("&gt", ">").replace("&lt", "<")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("≤", "<=").replace("≥", ">=")
    value = re.sub(r"[\u200b-\u200d\ufeff]", "", value)
    value = YEAR_SUFFIX_RE.sub(" ", value)
    value = COUNT_SUFFIX_RE.sub("", value)
    value = DAY_ORDINAL_SUFFIX_RE.sub("", value)
    value = re.sub(r"[^0-9a-z\u0980-\u09ff@%+\-√π°<>=.()\[\]{}/,∞]+", " ", value)
    return re.sub(r"\s+", "", value).strip(".")


def number_signature(text: object) -> tuple[str, ...]:
    output = []
    for value in NUMBER_RE.findall(canonical_answer(text)):
        try:
            normalized = format(Decimal(value).normalize(), "f")
            output.append("0" if normalized in {"-0", ""} else normalized)
        except InvalidOperation:
            output.append(value)
    return tuple(output)


def _as_answers(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [str(value).strip()] if str(value).strip() else []


def _answer_index(value: object, choices: list[str]) -> int | None:
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return ord(text) - ord("A")
    if text in {"0", "1", "2", "3"}:
        return int(text)
    if text in {"1", "2", "3", "4"}:
        return int(text) - 1
    normalized = canonical_answer(value)
    normalized_choices = [canonical_answer(choice) for choice in choices]
    return normalized_choices.index(normalized) if normalized in normalized_choices else None


def load_suman(path: Path) -> list[BankEntry]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_repo") != "SumanMondal/bengali_qa":
                continue
            answers = _as_answers(row.get("answer", ""))
            if answers:
                entries.append(
                    BankEntry(
                        question=str(row["question"]),
                        answers=answers,
                        choices=[],
                        correct_index=None,
                        source_repo="SumanMondal/bengali_qa",
                        source_file=str(row.get("source_file", "")),
                        license="apache-2.0",
                        source_kind="open_qa",
                    )
                )
    return entries


def load_hishab(cache_dir: Path) -> list[BankEntry]:
    frames = []
    for file_name in HISHAB_FILES:
        local = hf_hub_download(
            repo_id=HISHAB_REPO,
            repo_type="dataset",
            filename=file_name,
            cache_dir=cache_dir,
        )
        frame = pd.read_parquet(local)
        frame["_source_file"] = file_name
        frames.append(frame)
    entries = []
    for row in pd.concat(frames, ignore_index=True).to_dict("records"):
        choices = [str(item) for item in list(row["choices"])]
        index = _answer_index(row["answer"], choices)
        if index is None or not 0 <= index < len(choices):
            continue
        entries.append(
            BankEntry(
                question=str(row["question"]),
                answers=[choices[index]],
                choices=choices,
                correct_index=index,
                source_repo=HISHAB_REPO,
                source_file=str(row["_source_file"]),
                license="apache-2.0",
                source_kind="mcq",
            )
        )
    return entries


def ensure_bluck(bluck_dir: Path | None, cache_dir: Path) -> Path | None:
    if bluck_dir and bluck_dir.exists():
        return bluck_dir
    archive = cache_dir / "bluck-main.zip"
    extracted = cache_dir / "bluck-main"
    if extracted.exists():
        return extracted
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BLUCK_ZIP, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cache_dir)
        return extracted
    except Exception as exc:
        print("BLUCK unavailable", repr(exc))
        return None


def load_bluck(path: Path | None) -> list[BankEntry]:
    if path is None:
        return []
    entries = []
    for file_name in sorted((path / "bn_dataset").rglob("*.csv")):
        frame = pd.read_csv(file_name)
        for row in frame.itertuples():
            choices = [str(row.a), str(row.b), str(row.c), str(row.d)]
            index = _answer_index(row.answer, choices)
            if index is None:
                continue
            entries.append(
                BankEntry(
                    question=str(row.question),
                    answers=[choices[index]],
                    choices=choices,
                    correct_index=index,
                    source_repo="minhaj1403/bluck",
                    source_file=str(file_name.relative_to(path)),
                    license="cc-by-sa-4.0",
                    source_kind="mcq",
                )
            )
    return entries


def load_somaj(cache_dir: Path) -> list[BankEntry]:
    try:
        local = hf_hub_download(
            repo_id=SOMAJ_REPO,
            repo_type="dataset",
            filename=SOMAJ_FILE,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        print("SOMAJGYAAN unavailable", repr(exc))
        return []
    frame = pd.read_csv(local)
    entries = []
    for row in frame.itertuples():
        choices = [str(row.A), str(row.B), str(row.C), str(row.D)]
        index = _answer_index(row.Answer, choices)
        if index is None:
            continue
        entries.append(
            BankEntry(
                question=str(row.Question),
                answers=[choices[index]],
                choices=choices,
                correct_index=index,
                source_repo=SOMAJ_REPO,
                source_file=SOMAJ_FILE,
                license="mit",
                source_kind="mcq",
            )
        )
    return entries


def deduplicate(entries: Iterable[BankEntry]) -> list[BankEntry]:
    seen = set()
    output = []
    for entry in entries:
        key = (
            normalize_bn(entry.question),
            tuple(canonical_answer(answer) for answer in entry.answers),
            tuple(canonical_answer(choice) for choice in entry.choices),
            entry.correct_index,
            entry.source_repo,
        )
        if key not in seen:
            seen.add(key)
            output.append(entry)
    return output


def write_bank(entries: list[BankEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    print("wrote", path, "entries", len(entries))


def read_bank(path: Path) -> list[BankEntry]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(BankEntry(**json.loads(line)))
    return entries


def compare_mcq(response: str, entry: BankEntry) -> Decision | None:
    if entry.correct_index is None or not entry.choices:
        return None
    response_norm = canonical_answer(response)
    choices = [canonical_answer(choice) for choice in entry.choices]
    gold_index = int(entry.correct_index)
    gold = choices[gold_index]
    if not response_norm or not gold:
        return None

    response_structural = structural_answer(response)
    structural_choices = [structural_answer(choice) for choice in entry.choices]
    exact = [i for i, choice in enumerate(structural_choices) if response_structural == choice]
    if exact:
        label = int(gold_index in exact)
        return Decision(label, 1.0, "mcq_exact_option", entry.source_repo, entry.answers[0])

    response_numbers = number_signature(response_norm)
    gold_numbers = number_signature(gold)
    if response_numbers and gold_numbers and response_numbers != gold_numbers:
        matching_choices = [i for i, choice in enumerate(choices) if number_signature(choice) == response_numbers]
        if len(matching_choices) == 1:
            label = int(matching_choices[0] == gold_index)
            return Decision(label, 0.995, "mcq_number_option", entry.source_repo, entry.answers[0])
        return Decision(0, 0.99, "mcq_number_mismatch", entry.source_repo, entry.answers[0])
    if response_numbers and response_numbers == gold_numbers:
        return Decision(1, 0.99, "mcq_number_match", entry.source_repo, entry.answers[0])

    # Check for a near-exact explicit distractor before gold containment. A
    # distractor such as "Both physical and logical addresses" contains the
    # gold phrase "Logical address" but is still the wrong MCQ option.
    scores = [SequenceMatcher(None, response_norm, choice).ratio() for choice in choices]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best, runner_up = order[0], order[1]
    margin = scores[best] - scores[runner_up]
    if best != gold_index and scores[best] >= 0.90 and margin >= 0.08:
        return Decision(
            0,
            min(0.99, 0.70 + scores[best] * 0.25),
            "mcq_fuzzy_distractor",
            entry.source_repo,
            entry.answers[0],
        )

    if min(len(response_norm), len(gold)) >= 3 and (response_norm in gold or gold in response_norm):
        return Decision(1, 0.98, "mcq_gold_containment", entry.source_repo, entry.answers[0])

    if scores[best] >= 0.72 and margin >= 0.08:
        label = int(best == gold_index)
        return Decision(label, min(0.97, 0.70 + scores[best] * 0.25), "mcq_fuzzy_option", entry.source_repo, entry.answers[0])
    return None


def compare_open_qa(response: str, entry: BankEntry) -> Decision | None:
    response_norm = canonical_answer(response)
    golds = [canonical_answer(answer) for answer in entry.answers if canonical_answer(answer)]
    if not response_norm or not golds:
        return None
    if response_norm in golds:
        return Decision(1, 1.0, "qa_exact", entry.source_repo, " | ".join(entry.answers))
    response_numbers = number_signature(response_norm)
    gold_numbers = {number for gold in golds for number in number_signature(gold)}
    if response_numbers and gold_numbers and set(response_numbers) != gold_numbers:
        if any(len(number.replace(".", "")) > 6 for number in gold_numbers):
            return None
        scale_tokens = set(response_norm.split()) | {token for gold in golds for token in gold.split()}
        if not (scale_tokens & SCALE_WORDS):
            return Decision(0, 0.99, "qa_number_mismatch", entry.source_repo, " | ".join(entry.answers))
        return None
    if gold_numbers and not response_numbers:
        # Bengali number words (e.g. "দুই লক্ষ ...") need the semantic judge;
        # lexical mismatch is not evidence that the value is wrong.
        return None

    response_tokens = set(response_norm.split())
    gold_tokens = [set(gold.split()) for gold in golds]
    best_jaccard = max(
        len(response_tokens & tokens) / max(1, len(response_tokens | tokens)) for tokens in gold_tokens
    )
    if best_jaccard < 0.35 and len(response_tokens) >= 2:
        response_has_bn = bool(re.search(r"[\u0980-\u09ff]", response_norm))
        response_has_latin = bool(re.search(r"[a-z]", response_norm))
        gold_has_bn = any(re.search(r"[\u0980-\u09ff]", gold) for gold in golds)
        gold_has_latin = any(re.search(r"[a-z]", gold) for gold in golds)
        if (response_has_bn and gold_has_latin and not gold_has_bn) or (
            response_has_latin and gold_has_bn and not response_has_bn
        ):
            # A lexical mismatch cannot judge translations/transliterations such
            # as "গণবিধ্বংসী অস্ত্র" vs "Weapons of Mass Destruction".
            return None
        return Decision(0, 0.90, "qa_text_mismatch", entry.source_repo, " | ".join(entry.answers))
    return None


def compare_response(response: str, entry: BankEntry) -> Decision | None:
    return compare_mcq(response, entry) if entry.source_kind == "mcq" else compare_open_qa(response, entry)


def index_bank(entries: list[BankEntry]) -> dict[str, list[BankEntry]]:
    index: dict[str, list[BankEntry]] = defaultdict(list)
    for entry in entries:
        index[normalize_bn(entry.question)].append(entry)
    return dict(index)


def _answers_compatible(left: str, right: str) -> bool:
    left_norm, right_norm = canonical_answer(left), canonical_answer(right)
    if not left_norm or not right_norm:
        return True
    if left_norm == right_norm or (
        min(len(left_norm), len(right_norm)) >= 3
        and (left_norm in right_norm or right_norm in left_norm)
    ):
        return True
    left_numbers, right_numbers = number_signature(left_norm), number_signature(right_norm)
    if left_numbers and right_numbers and left_numbers == right_numbers:
        return True
    left_tokens, right_tokens = set(left_norm.split()), set(right_norm.split())
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return jaccard >= 0.5


def source_answers_conflict(entries: list[BankEntry]) -> bool:
    by_source: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        by_source[entry.source_repo].extend(entry.answers)
    sources = list(by_source)
    for i, left_source in enumerate(sources):
        for right_source in sources[i + 1 :]:
            if not any(
                _answers_compatible(left, right)
                for left in by_source[left_source]
                for right in by_source[right_source]
            ):
                return True
    return False


def decide_row(response: str, entries: list[BankEntry]) -> tuple[Decision | None, list[Decision]]:
    decisions = [decision for entry in entries if (decision := compare_response(response, entry)) is not None]
    if source_answers_conflict(entries):
        return None, decisions
    if not decisions:
        return None, []
    labels = {decision.label for decision in decisions if decision.confidence >= 0.90}
    if len(labels) != 1:
        return None, decisions
    candidates = [decision for decision in decisions if decision.label in labels]
    candidates.sort(key=lambda item: (item.confidence, item.source_repo != HISHAB_REPO), reverse=True)
    return candidates[0], decisions


def audit(df: pd.DataFrame, entries: list[BankEntry], split: str) -> tuple[list[dict], dict]:
    index = index_bank(entries)
    records = []
    predictions = []
    truths = []
    matched = 0
    for row in df.itertuples():
        sources = index.get(normalize_bn(row.prompt_bn), [])
        if not sources:
            continue
        matched += 1
        decision, all_decisions = decide_row(str(row.response_bn), sources)
        if decision is not None:
            predictions.append(decision.label)
            if "label" in df.columns:
                truths.append(int(row.label))
        records.append(
            {
                "id": int(row.id) if hasattr(row, "id") else int(row.Index),
                "decision": asdict(decision) if decision else None,
                "all_decisions": [asdict(item) for item in all_decisions],
                "sources": sorted({entry.source_repo for entry in sources}),
                "response": str(row.response_bn),
            }
        )
    summary = {
        "split": split,
        "rows": len(df),
        "matched_questions": matched,
        "confident_decisions": len(predictions),
        "abstained": matched - len(predictions),
        "source_match_counts": {
            source: sum(source in record["sources"] for record in records)
            for source in sorted({source for record in records for source in record["sources"]})
        },
    }
    if truths:
        summary["metrics"] = metric_row(pd.Series(truths).to_numpy(), pd.Series(predictions).to_numpy())
        summary["correct"] = sum(int(a == b) for a, b in zip(truths, predictions))
    return records, summary


def apply_bank(df: pd.DataFrame, base: pd.DataFrame, entries: list[BankEntry]) -> tuple[pd.DataFrame, list[dict]]:
    if len(df) != len(base) or not (df["id"].to_numpy() == base["id"].to_numpy()).all():
        raise ValueError("base submission IDs do not match split")
    output = base.copy()
    index = index_bank(entries)
    trace = []
    for row in df.itertuples():
        sources = index.get(normalize_bn(row.prompt_bn), [])
        decision, all_decisions = decide_row(str(row.response_bn), sources)
        if decision is None:
            continue
        mask = output["id"] == row.id
        old = int(output.loc[mask, "label"].iloc[0])
        output.loc[mask, "label"] = decision.label
        trace.append(
            {
                "id": int(row.id),
                "old": old,
                "new": decision.label,
                "changed": old != decision.label,
                "decision": asdict(decision),
                "all_decisions": [asdict(item) for item in all_decisions],
            }
        )
    return output, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and apply a public-source Bengali answer bank.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--external-qa", type=Path, default=OUTPUT_DIR / "external_qa_corpus.jsonl")
    build.add_argument("--cache-dir", type=Path, default=OUTPUT_DIR / "source_cache" / "answer_bank")
    build.add_argument("--bluck-dir", type=Path)
    build.add_argument("--output", type=Path, default=OUTPUT_DIR / "source_answer_bank.jsonl")

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--bank", type=Path, default=OUTPUT_DIR / "source_answer_bank.jsonl")
    audit_parser.add_argument("--split", choices=["sample", "test"], required=True)
    audit_parser.add_argument("--trace", type=Path)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--bank", type=Path, default=OUTPUT_DIR / "source_answer_bank.jsonl")
    apply_parser.add_argument("--base-submission", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--trace", type=Path, default=OUTPUT_DIR / "source_answer_bank_apply_trace.json")

    args = parser.parse_args()
    if args.command == "build":
        entries = []
        entries.extend(load_suman(args.external_qa))
        entries.extend(load_hishab(args.cache_dir / "huggingface"))
        entries.extend(load_bluck(ensure_bluck(args.bluck_dir, args.cache_dir / "bluck")))
        entries.extend(load_somaj(args.cache_dir / "huggingface"))
        entries = deduplicate(entries)
        write_bank(entries, args.output)
        counts = pd.Series([entry.source_repo for entry in entries]).value_counts().to_dict()
        print("sources", counts)
        return

    entries = read_bank(args.bank)
    if args.command == "audit":
        df = load_sample() if args.split == "sample" else load_test()
        records, summary = audit(df, entries, args.split)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print("wrote", args.trace)
        return

    df = load_test()
    base = pd.read_csv(args.base_submission)
    output, trace = apply_bank(df, base, entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.output, output["label"].value_counts().sort_index().to_dict())
    print("overrides", len(trace), "changed", sum(item["changed"] for item in trace))


if __name__ == "__main__":
    main()
