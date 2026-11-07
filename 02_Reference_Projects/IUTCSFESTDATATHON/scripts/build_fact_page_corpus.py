from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

from datathon_utils import normalize_bn
from wiki_evidence import fetch_extracts, get_json, search_pages


TOPIC_QUERIES = [
    "কাজী নজরুল ইসলাম",
    "জগদীশচন্দ্র বসু",
    "পদ্মা সেতু",
    "সুন্দরবন",
    "সাকিব আল হাসান",
    "ঢাকা মেট্রোরেল",
    "সত্যেন্দ্রনাথ বসু",
    "কেন্দ্রীয় শহীদ মিনার",
    "জাতীয় সংসদ ভবন",
    "কক্সবাজার সমুদ্র সৈকত",
    "ইনানী সমুদ্র সৈকত",
    "লালবাগ কেল্লা",
    "আহসান মঞ্জিল",
    "দেবদাস উপন্যাস",
    "আনন্দমঠ",
    "বন্দে মাতরম",
    "পথের পাঁচালী",
    "গোরা উপন্যাস",
    "কপালকুণ্ডলা",
    "শেষের কবিতা",
    "বিষবৃক্ষ",
    "রাজসিংহ উপন্যাস",
    "মেঘনাদবধ কাব্য",
    "শরৎচন্দ্র চট্টোপাধ্যায়",
    "পল্লীসমাজ",
    "বড়দিদি",
    "দুর্গেশনন্দিনী",
    "কৃষ্ণকান্তের উইল",
]


def existing_queries(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {item["query"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


def fetch_exact_title(query: str, chars: int) -> dict | None:
    data = get_json(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 0,
            "redirects": 1,
            "titles": query,
            "format": "json",
            "utf8": 1,
        },
        sleep=0.2,
    )
    pages = data.get("query", {}).get("pages", {})
    candidates = [page for page in pages.values() if "missing" not in page and page.get("extract")]
    if not candidates:
        return None
    page = max(candidates, key=lambda item: len(item.get("extract", "")))
    return {
        "pageid": int(page["pageid"]),
        "title": page.get("title", ""),
        "extract": page.get("extract", "")[:chars],
        "search_hits": [],
    }


def title_score(query: str, title: str) -> float:
    left, right = normalize_bn(query), normalize_bn(title)
    if left == right:
        return 2.0
    containment = 1.0 if left in right or right in left else 0.0
    return containment + SequenceMatcher(None, left, right).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact Bengali Wikipedia corpus for recurring fact topics.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-chars", type=int, default=16_000)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    records = existing_queries(args.output)
    for query in TOPIC_QUERIES:
        if query in records and records[query].get("extract"):
            continue
        record = {"query": query, "source": "bn.wikipedia.org", "license": "CC BY-SA"}
        for attempt in range(4):
            try:
                exact = fetch_exact_title(query, args.extract_chars)
                if exact:
                    record.update(exact)
                    break
                hits = search_pages(f'intitle:"{query}"', limit=5)
                if not hits:
                    hits = search_pages(query, limit=5)
                pageids = [int(hit["pageid"]) for hit in hits]
                extracts = fetch_extracts(pageids, chars=args.extract_chars)
                usable = [hit for hit in hits if extracts.get(int(hit["pageid"]), "")]
                if not usable:
                    record["error"] = "no_extracts"
                    break
                best = max(usable, key=lambda hit: title_score(query, hit.get("title", "")))
                pageid = int(best["pageid"])
                record.update({
                    "pageid": pageid,
                    "title": best.get("title", ""),
                    "extract": extracts.get(pageid, ""),
                    "search_hits": [
                        {"pageid": int(hit["pageid"]), "title": hit.get("title", "")}
                        for hit in hits
                    ],
                })
                break
            except Exception as exc:
                record["error"] = repr(exc)
                time.sleep(args.sleep * (attempt + 2))
        records[query] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(list(records.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(query, "->", record.get("title"), "chars", len(record.get("extract", "")), record.get("error", ""))
        time.sleep(args.sleep)

    print("wrote", args.output, "topics", len(records), "with_extract", sum(bool(x.get("extract")) for x in records.values()))


if __name__ == "__main__":
    main()
