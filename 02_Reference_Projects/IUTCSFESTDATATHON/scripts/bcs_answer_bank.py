from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from source_answer_bank import BankEntry, deduplicate, read_bank, write_bank


QUESTION_NUMBER_RE = re.compile(r"^\s*\d+\s*[.)।:-]\s*")
OPTION_PREFIX_RE = re.compile(r"^\s*(?:[A-Da-d]|[কখগঘ])\s*[.)।:-]\s*")
BN_OPTION_INDEX = {"ক": 0, "খ": 1, "গ": 2, "ঘ": 3}
QDOC_LINE_RE = re.compile(r"^\s*(?:[*-]\s*)?[০-৯0-9]+[.)।]\s*(.+?)\s+—\s+(.+?)\s*$")


def clean_question(value: object) -> str:
    return QUESTION_NUMBER_RE.sub("", str(value or "").strip())


def clean_option(value: object) -> str:
    return OPTION_PREFIX_RE.sub("", str(value or "").strip())


def iter_json(path: Path) -> Iterator[tuple[Path, object]]:
    for file_name in sorted(path.rglob("*.json")):
        try:
            yield file_name, json.loads(file_name.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def walk_question_dicts(value: object) -> Iterator[dict]:
    if isinstance(value, list):
        for item in value:
            yield from walk_question_dicts(item)
    elif isinstance(value, dict):
        if "question" in value and "options" in value and "answer" in value:
            yield value
        for key in ("questions", "subQuestions", "items"):
            if key in value:
                yield from walk_question_dicts(value[key])


def load_8020(path: Path) -> list[BankEntry]:
    entries = []
    for file_name, payload in iter_json(path):
        for row in walk_question_dicts(payload):
            options = row.get("options")
            if isinstance(options, dict):
                keys = [key for key in ("A", "B", "C", "D") if key in options]
                choices = [clean_option(options[key]) for key in keys]
                answer = str(row.get("answer", "")).strip().upper()
                index = keys.index(answer) if answer in keys else None
            elif isinstance(options, list):
                choices = [clean_option(item) for item in options]
                answer = str(row.get("answer", "")).strip().upper()
                index = ord(answer) - ord("A") if answer in {"A", "B", "C", "D"} else None
            else:
                continue
            question = clean_question(row.get("question"))
            if not question or index is None or not 0 <= index < len(choices) or not choices[index]:
                continue
            entries.append(
                BankEntry(
                    question=question,
                    answers=[choices[index]],
                    choices=choices,
                    correct_index=index,
                    source_repo="tanvirmahfuz100/80-20-exam",
                    source_file=str(file_name.relative_to(path)),
                    license="public-repository-no-declared-data-license",
                    source_kind="mcq",
                )
            )
    return entries


def load_farhan(path: Path) -> list[BankEntry]:
    entries = []
    for file_name, payload in iter_json(path):
        rows = payload.get("questions", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            choices = [clean_option(row.get(f"option_{letter}", "")) for letter in "abcd"]
            raw_answer = str(row.get("correct_answer", "")).strip()
            index = BN_OPTION_INDEX.get(raw_answer[:1])
            if index is None:
                cleaned_answer = clean_option(raw_answer)
                matches = [i for i, choice in enumerate(choices) if choice == cleaned_answer]
                index = matches[0] if len(matches) == 1 else None
            question = clean_question(row.get("question_text"))
            if not question or index is None or not 0 <= index < len(choices) or not choices[index]:
                continue
            entries.append(
                BankEntry(
                    question=question,
                    answers=[choices[index]],
                    choices=choices,
                    correct_index=index,
                    source_repo="mojnupust/api.farhan-mcq",
                    source_file=str(file_name.relative_to(path)),
                    license="mit",
                    source_kind="mcq",
                )
            )
    return entries


def load_job_mcq(path: Path) -> list[BankEntry]:
    entries = []
    for file_name, payload in iter_json(path):
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict) or not isinstance(row.get("answers"), list):
                continue
            answers = row["answers"]
            choices = [clean_option(item.get("text", "")) for item in answers if isinstance(item, dict)]
            correct = [i for i, item in enumerate(answers) if isinstance(item, dict) and item.get("score") == 1]
            index = correct[0] if len(correct) == 1 else None
            question = clean_question(row.get("question"))
            if not question or index is None or not 0 <= index < len(choices) or not choices[index]:
                continue
            entries.append(
                BankEntry(
                    question=question,
                    answers=[choices[index]],
                    choices=choices,
                    correct_index=index,
                    source_repo="NewtonGain/job_mcq",
                    source_file=str(file_name.relative_to(path)),
                    license="public-repository-no-declared-data-license",
                    source_kind="mcq",
                )
            )
    return entries


def load_qdoc(path: Path) -> list[BankEntry]:
    entries = []
    for file_name in sorted(path.rglob("*.topic")):
        if "QNA" not in str(file_name):
            continue
        for line in file_name.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = QDOC_LINE_RE.match(line)
            if not match:
                continue
            question, answer = (part.strip() for part in match.groups())
            if len(question) < 4 or not answer:
                continue
            entries.append(
                BankEntry(
                    question=question,
                    answers=[answer],
                    choices=[],
                    correct_index=None,
                    source_repo="Nh-em0n/QDoc",
                    source_file=str(file_name.relative_to(path)),
                    license="public-repository-no-declared-data-license",
                    source_kind="open_qa",
                )
            )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Add public BCS/job-exam answer keys to the source bank.")
    parser.add_argument("--base-bank", type=Path)
    parser.add_argument("--farhan-dir", type=Path)
    parser.add_argument("--exam-8020-dir", type=Path)
    parser.add_argument("--job-mcq-dir", type=Path)
    parser.add_argument("--qdoc-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries: list[BankEntry] = read_bank(args.base_bank) if args.base_bank else []
    before = len(entries)
    counts = {}
    loaders: Iterable[tuple[str, Path | None, object]] = (
        ("farhan", args.farhan_dir, load_farhan),
        ("80-20", args.exam_8020_dir, load_8020),
        ("job_mcq", args.job_mcq_dir, load_job_mcq),
        ("qdoc", args.qdoc_dir, load_qdoc),
    )
    for name, path, loader in loaders:
        if path:
            loaded = loader(path)
            counts[name] = len(loaded)
            entries.extend(loaded)
    entries = deduplicate(entries)
    write_bank(entries, args.output)
    print("base", before, "loaded", counts, "deduplicated_total", len(entries))


if __name__ == "__main__":
    main()
