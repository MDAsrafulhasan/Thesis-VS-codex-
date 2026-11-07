from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR, load_sample, load_test


API = "https://bn.wikipedia.org/w/api.php"


def clean_query(prompt: str, response: str = "") -> str:
    text = f"{prompt} {response}"
    text = re.sub(r"[\"“”‘’*_(){}\[\]:;!?।]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep query short; MediaWiki search gets noisy with long full prompts.
    words = text.split()
    return " ".join(words[:18])


def get_json(params: dict, sleep: float = 0.15) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": "IUTCSFESTDATATHON-hallucination-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    time.sleep(sleep)
    return data


def search_pages(query: str, limit: int = 3) -> list[dict]:
    data = get_json(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        }
    )
    return data.get("query", {}).get("search", [])


def fetch_extracts(pageids: list[int], chars: int = 1200) -> dict[int, str]:
    if not pageids:
        return {}
    data = get_json(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 0,
            "pageids": "|".join(str(x) for x in pageids),
            "format": "json",
            "utf8": 1,
        }
    )
    pages = data.get("query", {}).get("pages", {})
    out = {}
    for pid, page in pages.items():
        extract = page.get("extract", "")
        out[int(pid)] = extract[:chars]
    return out


def load_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["row_index"])] = row
    return rows


def build_evidence(df: pd.DataFrame, output: Path, limit: int | None = None, only_no_context: bool = True) -> None:
    output.parent.mkdir(exist_ok=True)
    existing = load_existing(output)
    rows = df[~df["has_context"]].copy() if only_no_context else df.copy()
    if limit is not None:
        rows = rows.head(limit)

    with output.open("a", encoding="utf-8") as handle:
        for row_index, row in rows.iterrows():
            if int(row_index) in existing:
                continue
            query = clean_query(row["prompt_bn"], row["response_bn"])
            try:
                hits = search_pages(query)
                extracts = fetch_extracts([int(hit["pageid"]) for hit in hits])
                evidence = [
                    {
                        "pageid": int(hit["pageid"]),
                        "title": hit.get("title", ""),
                        "snippet": re.sub("<.*?>", "", hit.get("snippet", "")),
                        "extract": extracts.get(int(hit["pageid"]), ""),
                    }
                    for hit in hits
                ]
                record = {
                    "row_index": int(row_index),
                    "id": int(row["id"]) if "id" in row else None,
                    "query": query,
                    "prompt_bn": row["prompt_bn"],
                    "response_bn": row["response_bn"],
                    "evidence": evidence,
                }
            except Exception as exc:
                record = {
                    "row_index": int(row_index),
                    "id": int(row["id"]) if "id" in row else None,
                    "query": query,
                    "prompt_bn": row["prompt_bn"],
                    "response_bn": row["response_bn"],
                    "error": repr(exc),
                    "evidence": [],
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print("wrote", row_index, query, "hits", len(record.get("evidence", [])), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["sample", "test"], default="test")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "wiki_evidence_test_noctx.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-rows", action="store_true")
    args = parser.parse_args()

    df = load_sample() if args.split == "sample" else load_test()
    build_evidence(df, Path(args.output), limit=args.limit, only_no_context=not args.all_rows)


if __name__ == "__main__":
    main()
