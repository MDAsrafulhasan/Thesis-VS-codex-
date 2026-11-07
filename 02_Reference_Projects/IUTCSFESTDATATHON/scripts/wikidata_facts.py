from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from datathon_utils import OUTPUT_DIR, clean_text


ENDPOINT = "https://query.wikidata.org/sparql"
SOURCE_REPO = "wikidata:query-service"
LICENSE = "CC0-1.0"
BN_RE = re.compile(r"[\u0980-\u09ff]")
DATE_RE = re.compile(r"^\d{3,4}(-\d{2}){0,2}$")
NUM_RE = re.compile(r"^\d+(\.\d+)?$")


QUERIES = {
    "bangladesh_properties": """
SELECT ?propName ?value ?valueLabel WHERE {
  VALUES (?prop ?propName) {
    (wdt:P36 "রাজধানী")
    (wdt:P37 "সরকারি ভাষা")
    (wdt:P38 "মুদ্রা")
    (wdt:P30 "মহাদেশ")
    (wdt:P150 "প্রশাসনিক বিভাগ")
  }
  wd:Q902 ?prop ?value.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "bn,en". }
}
LIMIT {limit}
""",
    "bangladesh_places": """
SELECT ?item ?itemLabel ?typeLabel ?adminLabel ?inception WHERE {
  ?item wdt:P17 wd:Q902.
  OPTIONAL { ?item wdt:P31 ?type. }
  OPTIONAL { ?item wdt:P131 ?admin. }
  OPTIONAL { ?item wdt:P571 ?inception. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "bn,en". }
}
LIMIT {limit}
""",
    "bangladesh_place_properties": """
SELECT ?item ?itemLabel ?propName ?value ?valueLabel WHERE {
  ?item wdt:P17 wd:Q902.
  VALUES (?prop ?propName) {
    (wdt:P2043 "দৈর্ঘ্য")
    (wdt:P2046 "আয়তন")
    (wdt:P1082 "জনসংখ্যা")
    (wdt:P571 "প্রতিষ্ঠাকাল")
    (wdt:P585 "তারিখ")
    (wdt:P580 "শুরুর সময়")
    (wdt:P582 "শেষ সময়")
    (wdt:P112 "প্রতিষ্ঠাতা")
    (wdt:P159 "প্রধান কার্যালয়")
    (wdt:P276 "অবস্থান")
    (wdt:P361 "অংশ")
  }
  ?item ?prop ?value.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "bn,en". }
}
LIMIT {limit}
""",
    "bangladeshi_people": """
SELECT ?item ?itemLabel ?occupationLabel ?birthPlaceLabel ?birthDate ?deathDate WHERE {
  ?item wdt:P27 wd:Q902.
  OPTIONAL { ?item wdt:P106 ?occupation. }
  OPTIONAL { ?item wdt:P19 ?birthPlace. }
  OPTIONAL { ?item wdt:P569 ?birthDate. }
  OPTIONAL { ?item wdt:P570 ?deathDate. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "bn,en". }
}
LIMIT {limit}
""",
}


def sparql(query: str, cache_path: Path, sleep_seconds: float) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "IUTCSFESTDATATHON evidence builder (local research; contact via Kaggle team)",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return payload


def value(binding: dict[str, Any], key: str) -> str:
    return clean_text((binding.get(key) or {}).get("value", ""))


def date_text(raw: str) -> str:
    raw = clean_text(raw)
    if not raw:
        return ""
    if "T" in raw:
        raw = raw.split("T", 1)[0]
    return raw.lstrip("+")


def fact_value(raw: str) -> str:
    raw = date_text(raw)
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw


def add_row(rows: list[dict], question: str, answer: str, context: str, source_file: str) -> None:
    question = clean_text(question)
    answer = clean_text(answer)
    context = clean_text(context)
    if not question or not answer:
        return
    if not BN_RE.search(answer) and not DATE_RE.search(answer) and not NUM_RE.search(answer):
        return
    rows.append(
        {
            "question": question,
            "answer": answer,
            "context": context,
            "source_repo": SOURCE_REPO,
            "source_file": source_file,
            "license": LICENSE,
        }
    )


def convert_bangladesh_properties(bindings: list[dict[str, Any]]) -> list[dict]:
    rows: list[dict] = []
    for item in bindings:
        prop = value(item, "propName")
        answer = value(item, "valueLabel")
        context = f"উইকিডাটা অনুযায়ী বাংলাদেশ সম্পর্কিত তথ্য: {prop}: {answer}।"
        add_row(rows, f"বাংলাদেশের {prop} কী?", answer, context, "bangladesh_properties")
        add_row(rows, f"বাংলাদেশের {prop} কোনটি?", answer, context, "bangladesh_properties")
    return rows


def convert_places(bindings: list[dict[str, Any]]) -> list[dict]:
    rows: list[dict] = []
    for item in bindings:
        name = value(item, "itemLabel")
        type_name = value(item, "typeLabel")
        admin = value(item, "adminLabel")
        inception = date_text(value(item, "inception"))
        context_parts = [f"উইকিডাটা সত্তা: {name}।", "দেশ: বাংলাদেশ।"]
        if type_name:
            context_parts.append(f"ধরন: {type_name}।")
        if admin:
            context_parts.append(f"প্রশাসনিক অঞ্চল: {admin}।")
        if inception:
            context_parts.append(f"প্রতিষ্ঠাকাল: {inception}।")
        context = " ".join(context_parts)
        add_row(rows, f'"{name}" কোন দেশে অবস্থিত?', "বাংলাদেশ", context, "bangladesh_places")
        if type_name:
            add_row(rows, f'"{name}" কোন ধরনের সত্তা?', type_name, context, "bangladesh_places")
        if admin and admin != name:
            add_row(rows, f'"{name}" কোন প্রশাসনিক অঞ্চলে অবস্থিত?', admin, context, "bangladesh_places")
        if inception:
            add_row(rows, f'"{name}" এর প্রতিষ্ঠাকাল কবে?', inception, context, "bangladesh_places")
    return rows


def property_question(name: str, prop: str) -> str:
    if prop == "দৈর্ঘ্য":
        return f'"{name}" এর দৈর্ঘ্য কত?'
    if prop == "আয়তন":
        return f'"{name}" এর আয়তন কত?'
    if prop == "জনসংখ্যা":
        return f'"{name}" এর জনসংখ্যা কত?'
    if prop == "প্রতিষ্ঠাকাল":
        return f'"{name}" এর প্রতিষ্ঠাকাল কবে?'
    if prop == "তারিখ":
        return f'"{name}" কবে অনুষ্ঠিত হয়?'
    if prop == "শুরুর সময়":
        return f'"{name}" কবে শুরু হয়?'
    if prop == "শেষ সময়":
        return f'"{name}" কবে শেষ হয়?'
    if prop == "প্রতিষ্ঠাতা":
        return f'"{name}" এর প্রতিষ্ঠাতা কে?'
    if prop == "প্রধান কার্যালয়":
        return f'"{name}" এর প্রধান কার্যালয় কোথায়?'
    if prop == "অবস্থান":
        return f'"{name}" কোথায় অবস্থিত?'
    if prop == "অংশ":
        return f'"{name}" কোনটির অংশ?'
    return f'"{name}" এর {prop} কী?'


def convert_place_properties(bindings: list[dict[str, Any]]) -> list[dict]:
    rows: list[dict] = []
    for item in bindings:
        name = value(item, "itemLabel")
        prop = value(item, "propName")
        answer = fact_value(value(item, "valueLabel") or value(item, "value"))
        context = f"উইকিডাটা সত্তা: {name}। দেশ: বাংলাদেশ। {prop}: {answer}।"
        add_row(rows, property_question(name, prop), answer, context, "bangladesh_place_properties")
    return rows


def convert_people(bindings: list[dict[str, Any]]) -> list[dict]:
    rows: list[dict] = []
    for item in bindings:
        name = value(item, "itemLabel")
        occupation = value(item, "occupationLabel")
        birth_place = value(item, "birthPlaceLabel")
        birth_date = date_text(value(item, "birthDate"))
        death_date = date_text(value(item, "deathDate"))
        context_parts = [f"উইকিডাটা ব্যক্তি: {name}।", "নাগরিকত্ব: বাংলাদেশ।"]
        if occupation:
            context_parts.append(f"পেশা: {occupation}।")
        if birth_place:
            context_parts.append(f"জন্মস্থান: {birth_place}।")
        if birth_date:
            context_parts.append(f"জন্মতারিখ: {birth_date}।")
        if death_date:
            context_parts.append(f"মৃত্যুতারিখ: {death_date}।")
        context = " ".join(context_parts)
        add_row(rows, f'"{name}" কোন দেশের নাগরিক?', "বাংলাদেশ", context, "bangladeshi_people")
        if occupation:
            add_row(rows, f'"{name}" এর পেশা কী?', occupation, context, "bangladeshi_people")
        if birth_place:
            add_row(rows, f'"{name}" কোথায় জন্মগ্রহণ করেন?', birth_place, context, "bangladeshi_people")
        if birth_date:
            add_row(rows, f'"{name}" কবে জন্মগ্রহণ করেন?', birth_date, context, "bangladeshi_people")
        if death_date:
            add_row(rows, f'"{name}" কবে মৃত্যুবরণ করেন?', death_date, context, "bangladeshi_people")
    return rows


CONVERTERS = {
    "bangladesh_properties": convert_bangladesh_properties,
    "bangladesh_places": convert_places,
    "bangladesh_place_properties": convert_place_properties,
    "bangladeshi_people": convert_people,
}


def query_to_rows(name: str, raw: dict[str, Any]) -> list[dict]:
    bindings = raw.get("results", {}).get("bindings", [])
    return CONVERTERS[name](bindings)


def write_corpus(rows: list[dict], output: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).fillna("")
    df = df.drop_duplicates(["question", "answer", "context"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    df.to_csv(output.with_suffix(".csv"), index=False)
    print("wrote", output.with_suffix(".jsonl"), output.with_suffix(".csv"), "rows", len(df))
    print(df["source_file"].value_counts().head(20).to_string())
    return df


def make_pairs(corpus: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for source_file, group in corpus.groupby("source_file"):
        answers = group["answer"].drop_duplicates().tolist()
        if len(answers) < 2:
            continue
        for i, item in enumerate(group.to_dict("records")):
            rows.append(
                {
                    "context": item["context"],
                    "prompt_bn": item["question"],
                    "response_bn": item["answer"],
                    "label": 1,
                    "source": "wikidata_faithful",
                    "source_repo": item["source_repo"],
                }
            )
            wrong = answers[(i * 7919 + 17) % len(answers)]
            if wrong == item["answer"]:
                wrong = answers[(i * 7919 + 18) % len(answers)]
            rows.append(
                {
                    "context": item["context"],
                    "prompt_bn": item["question"],
                    "response_bn": wrong,
                    "label": 0,
                    "source": "wikidata_answer_swap",
                    "source_repo": item["source_repo"],
                }
            )
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    df.to_csv(output.with_suffix(".csv"), index=False)
    print("wrote", output.with_suffix(".jsonl"), output.with_suffix(".csv"), "rows", len(df))
    print(df["label"].value_counts().sort_index().to_string())
    return df


def merge_corpora(paths: list[Path], output: Path) -> None:
    frames = [pd.read_json(path, lines=True) for path in paths if path.exists()]
    if not frames:
        raise ValueError("no corpora to merge")
    write_corpus(pd.concat(frames, ignore_index=True).to_dict("records"), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default=str(OUTPUT_DIR / "wikidata_bangladesh_corpus"))
    parser.add_argument("--pairs-prefix", default=str(OUTPUT_DIR / "wikidata_bangladesh_pairs"))
    parser.add_argument("--cache-dir", default=str(OUTPUT_DIR / "source_cache" / "wikidata"))
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--merge-with", action="append", default=[])
    parser.add_argument("--merged-output-prefix", default=str(OUTPUT_DIR / "merged_evidence_corpus"))
    args = parser.parse_args()

    rows: list[dict] = []
    cache_dir = Path(args.cache_dir)
    for name, template in QUERIES.items():
        cache_path = cache_dir / f"{name}_{args.limit}.json"
        if args.skip_download and not cache_path.exists():
            print("skip missing cache", cache_path)
            continue
        raw = sparql(template.replace("{limit}", str(args.limit)), cache_path, args.sleep)
        query_rows = query_to_rows(name, raw)
        rows.extend(query_rows)
        print(name, "bindings", len(raw.get("results", {}).get("bindings", [])), "rows", len(query_rows))

    corpus = write_corpus(rows, Path(args.output_prefix))
    make_pairs(corpus, Path(args.pairs_prefix))
    merge_paths = [Path(x) for x in args.merge_with] + [Path(args.output_prefix).with_suffix(".jsonl")]
    if args.merge_with:
        merge_corpora(merge_paths, Path(args.merged_output_prefix))


if __name__ == "__main__":
    main()
