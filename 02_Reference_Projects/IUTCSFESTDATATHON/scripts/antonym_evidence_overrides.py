from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
import urllib.parse
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from datathon_utils import load_sample, load_test, normalize_bn


LIST_SOURCES = {
    "banglaquiz": "https://www.banglaquiz.in/2021/08/17/list-of-antonyms-in-bengali/",
    "bornoporichoy": "https://bornoporichoy.com/opposite-words-list-in-bengali/",
    "tetchallenger": "https://www.tetchallenger.com/2020/12/antonyms.html",
    "netexplanations": "https://www.netexplanations.com/antonyms-in-bengali/",
    "banglamcq": "https://www.banglamcq.in/opposite-words-list-in-bengali/",
    "wbnotes": "https://wbnotes.in/biporit-sobdo-bengali-grammar/",
    "dailymodeltest": "https://dailymodeltest.com/2024/01/11/bengali-antonyms/",
}

SUPPLEMENTAL_SOURCES = {
    "sattacademy_bhut": {
        "url": "https://sattacademy.com/mcq/%E0%A6%AD%E0%A7%82%E0%A6%A4-%E0%A6%B6%E0%A6%AC%E0%A7%8D%E0%A6%A6%E0%A7%87%E0%A6%B0-%E0%A6%AC%E0%A6%BF%E0%A6%AA%E0%A6%B0%E0%A7%80%E0%A6%A4-%E0%A6%85%E0%A6%B0%E0%A7%8D%E0%A6%A5",
        "headword": "ভূত",
        "pattern": r"ভূত এর বিপরীত শব্দ হলো\s*-\s*([^।<]+)",
    },
    "mcqacademy_tikkhno": {
        "url": "https://mcqacademy.com/bn/mcq/%E0%A6%A4%E0%A7%80%E0%A6%95%E0%A7%8D%E0%A6%B7%E0%A7%8D%E0%A6%A3-%E0%A6%8F%E0%A6%B0-%E0%A6%AC%E0%A6%BF%E0%A6%AA%E0%A6%B0%E0%A7%80%E0%A6%A4-%E0%A6%B6%E0%A6%AC%E0%A7%8D%E0%A6%A6-%E0%A6%95%E0%A7%80/",
        "headword": "তীক্ষ্ণ",
        "pattern": r"তীক্ষ্ণ[^।]{0,80}বিপরীত শব্দ হলো\s*([^।<]+)",
    },
}

WIKTIONARY_API = "https://bn.wiktionary.org/w/api.php"
WIKTIONARY_SOURCE = "bn_wiktionary"
WIKTIONARY_URL = "https://bn.wiktionary.org/wiki/{title}"


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "IUTCSFESTDATATHON-hallucination-research/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_json(url: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "IUTCSFESTDATATHON-hallucination-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def canonical(value: str) -> str:
    # ভবিষ্যত/ভবিষ্যৎ and similarৎ/ত spelling variants are equivalent for
    # source voting; punctuation and whitespace are already normalized.
    return normalize_bn(html.unescape(value)).replace("ৎ", "ত")


def split_answers(value: str) -> set[str]:
    return {
        canonical(part)
        for part in re.split(r"[/,;]| অথবা | বা ", value)
        if canonical(part)
    }


def parse_list_source(body: str) -> list[tuple[str, set[str]]]:
    parser = TableParser()
    parser.feed(body)
    pairs = []
    for row in parser.rows:
        if len(row) >= 3 and not row[0].strip("০১২৩৪৫৬৭৮৯0123456789. "):
            word, answer = row[1], row[2]
        elif len(row) == 2:
            word, answer = row
        else:
            continue
        if "শব্দ" in word and "বিপরীত" in answer:
            continue
        answers = split_answers(answer)
        if canonical(word) and answers:
            pairs.append((canonical(word), answers))
    return pairs


def parse_wiktionary_antonyms(wikitext: str) -> set[str]:
    """Extract plain Bengali values from a Bengali Wiktionary antonym section."""
    lines = wikitext.splitlines()
    answers: list[str] = []
    section_level = None
    collecting = False
    for raw in lines:
        heading = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", raw)
        if heading:
            level = len(heading.group(1))
            title = canonical(heading.group(2))
            if collecting and section_level is not None and level <= section_level:
                break
            if "বিপরীত" in title:
                collecting = True
                section_level = level
            continue
        if not collecting:
            continue
        value = re.sub(r"<!--.*?-->", " ", raw)
        value = re.sub(r"\{\{(?:l|link)\|(?:bn|as)\|([^{}|]+)(?:\|[^{}]*)?\}\}", r"\1", value)
        value = re.sub(r"\{\{[^{}]*\}\}", " ", value)
        value = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", value)
        value = re.sub(r"^[*#:;\s]+", "", value).strip(" ।")
        answers.extend(part for part in re.split(r"[,;/]| অথবা | বা ", value) if part.strip())
    return {canonical(answer) for answer in answers if canonical(answer)}


def wiktionary_records(headwords: list[str]) -> list[dict]:
    records = []
    unique = sorted({canonical(word) for word in headwords if canonical(word)})
    for offset in range(0, len(unique), 40):
        payload = fetch_json(
            WIKTIONARY_API,
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(unique[offset : offset + 40]),
                "format": "json",
                "formatversion": "2",
            },
        )
        for page in payload.get("query", {}).get("pages", []):
            if page.get("missing") or not page.get("revisions"):
                continue
            wikitext = page["revisions"][0].get("slots", {}).get("main", {}).get("content", "")
            answers = parse_wiktionary_antonyms(wikitext)
            if answers:
                title = str(page["title"])
                records.append(
                    {
                        "source": WIKTIONARY_SOURCE,
                        "url": WIKTIONARY_URL.format(title=urllib.parse.quote(title)),
                        "headword": canonical(title),
                        "answers": sorted(answers),
                    }
                )
    return records


def build_artifact(output: Path) -> None:
    records = []
    for source, url in LIST_SOURCES.items():
        body = fetch(url)
        for headword, answers in parse_list_source(body):
            records.append({"source": source, "url": url, "headword": headword, "answers": sorted(answers)})
        print(source, "pairs", sum(item["source"] == source for item in records))
    for source, config in SUPPLEMENTAL_SOURCES.items():
        try:
            body = html.unescape(fetch(config["url"]))
        except Exception as exc:
            # These are only supplemental single-question pages. A removed or
            # rate-limited page must not prevent rebuilding the licensed list
            # and Wiktionary evidence artifact.
            print(source, "unavailable", repr(exc))
            continue
        plain = re.sub(r"<[^>]+>", " ", body)
        plain = " ".join(plain.split())
        match = re.search(config["pattern"], plain)
        if not match:
            raise ValueError(f"could not parse supplemental source {source}")
        records.append(
            {
                "source": source,
                "url": config["url"],
                "headword": canonical(config["headword"]),
                "answers": sorted(split_answers(match.group(1))),
            }
        )
        print(source, records[-1]["headword"], records[-1]["answers"])
    headwords = [
        headword
        for prompt in load_test()["prompt_bn"].astype(str)
        if (headword := headword_from_prompt(prompt))
    ]
    wiki_records = wiktionary_records(headwords)
    records.extend(wiki_records)
    print(WIKTIONARY_SOURCE, "pairs", len(wiki_records))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", output, "records", len(records))


def headword_from_prompt(prompt: str) -> str | None:
    if "বিপরীত শব্দ" not in prompt and "বিপরীতার্থক শব্দ" not in prompt:
        return None
    match = re.search(r"[‘'\"]([^’'\"]+)[’'\"]", prompt)
    if match:
        return canonical(match.group(1))
    # The synthetic block consistently has a closing ASCII apostrophe but no
    # opening quote (e.g. ``সমীপ' শব্দের ...``).
    match = re.match(r"^([^'’\"]+)[’'\"]", prompt)
    return canonical(match.group(1)) if match else None


def source_consensus(
    headword: str,
    response: str,
    records: list[dict],
    min_sources: int,
    reject_alternatives: bool = False,
) -> tuple[int | None, dict]:
    by_source: dict[str, set[str]] = defaultdict(set)
    urls = {}
    for item in records:
        if item["headword"] == headword:
            by_source[item["source"]].update(item["answers"])
            urls[item["source"]] = item["url"]
    response_norm = canonical(response)
    supporters = [source for source, answers in by_source.items() if response_norm in answers]
    answer_votes: dict[str, set[str]] = defaultdict(set)
    for source, answers in by_source.items():
        for answer in answers:
            answer_votes[answer].add(source)
    consensus_answers = {answer: sorted(sources) for answer, sources in answer_votes.items() if len(sources) >= min_sources}
    if len(supporters) >= min_sources:
        prediction = 1
    elif reject_alternatives and consensus_answers and response_norm not in consensus_answers:
        # Unsafe for open lexical questions unless explicitly requested: a
        # headword can have several valid antonyms, while a source lists only
        # one (e.g. ক্ষণস্থায়ী -> দীর্ঘস্থায়ী / চিরস্থায়ী).
        prediction = 0
    else:
        prediction = None
    return prediction, {
        "headword": headword,
        "response": response,
        "supporters": supporters,
        "consensus_answers": consensus_answers,
        "source_answers": {source: sorted(answers) for source, answers in by_source.items()},
        "urls": urls,
    }


def apply_artifact(
    split: str,
    artifact: Path,
    base_submission: Path | None,
    output: Path | None,
    trace: Path,
    min_sources: int,
    reject_alternatives: bool = False,
) -> None:
    df = load_sample() if split == "sample" else load_test()
    records = json.loads(artifact.read_text(encoding="utf-8"))
    base = pd.read_csv(base_submission) if base_submission else None
    decisions = []
    for row_index, row in df.iterrows():
        headword = headword_from_prompt(str(row.prompt_bn))
        if not headword:
            continue
        pred, evidence = source_consensus(
            headword,
            str(row.response_bn),
            records,
            min_sources,
            reject_alternatives=reject_alternatives,
        )
        if pred is None:
            continue
        item = {"row_index": int(row_index), "id": int(row.id) if "id" in df.columns else int(row_index), "prediction": pred, **evidence}
        if "label" in df.columns:
            item.update({"truth": int(row.label), "correct": pred == int(row.label)})
        if base is not None:
            mask = base["id"] == int(row.id)
            old = int(base.loc[mask, "label"].iloc[0])
            item.update({"old": old, "changed": old != pred})
            base.loc[mask, "label"] = pred
        decisions.append(item)
    summary = {
        "split": split,
        "min_sources": min_sources,
        "reject_alternatives": reject_alternatives,
        "decisions": len(decisions),
        "correct": sum(item.get("correct", False) for item in decisions) if "label" in df.columns else None,
        "changed": sum(item.get("changed", False) for item in decisions) if base is not None else None,
    }
    if base is not None:
        if output is None:
            raise ValueError("--base-submission requires --output")
        base.to_csv(output, index=False)
        summary.update({"output": str(output), "output_counts": base["label"].value_counts().sort_index().to_dict()})
    trace.write_text(json.dumps({"summary": summary, "records": decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for item in decisions:
        if item.get("changed"):
            print("CHANGE", item["id"], item["old"], "->", item["prediction"], item["headword"], item["response"], item["consensus_answers"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and apply multi-source Bengali antonym evidence.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--split", choices=["sample", "test"], required=True)
    apply.add_argument("--artifact", type=Path, required=True)
    apply.add_argument("--base-submission", type=Path)
    apply.add_argument("--output", type=Path)
    apply.add_argument("--trace", type=Path, required=True)
    apply.add_argument("--min-sources", type=int, default=2)
    apply.add_argument(
        "--reject-alternatives",
        action="store_true",
        help="Legacy unsafe mode: reject a response merely because sources agree on another antonym.",
    )
    args = parser.parse_args()
    if args.command == "build":
        build_artifact(args.output)
    else:
        apply_artifact(
            args.split,
            args.artifact,
            args.base_submission,
            args.output,
            args.trace,
            args.min_sources,
            reject_alternatives=args.reject_alternatives,
        )


if __name__ == "__main__":
    main()
