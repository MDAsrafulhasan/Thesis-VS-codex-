from __future__ import annotations

import argparse
import html
import json
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from build_fact_page_corpus import TOPIC_QUERIES


class ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.parts: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "p":
            self.depth += 1
            if self.depth == 1:
                self.parts = []
        elif self.depth and tag in {"br", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self.depth:
            self.depth -= 1
            if self.depth == 0:
                text = " ".join("".join(self.parts).split())
                if text:
                    self.paragraphs.append(text)
                self.parts = []

    def handle_data(self, data: str) -> None:
        if self.depth:
            self.parts.append(data)


def fetch_page(title: str, chars: int) -> tuple[str, str]:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="()")
    request = urllib.request.Request(
        f"https://bn.wikipedia.org/wiki/{encoded}",
        headers={"User-Agent": "IUTCSFESTDATATHON-hallucination-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
        final_title = urllib.parse.unquote(response.url.rsplit("/", 1)[-1]).replace("_", " ")
    parser = ParagraphParser()
    parser.feed(body)
    extract = "\n".join(html.unescape(paragraph) for paragraph in parser.paragraphs)
    return final_title, extract[:chars]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact Bengali Wikipedia page extracts from article HTML.")
    parser.add_argument("--api-corpus", type=Path, required=True, help="API corpus used for resolved canonical titles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-chars", type=int, default=30_000)
    parser.add_argument("--sleep", type=float, default=0.6)
    args = parser.parse_args()

    api_records = {item["query"]: item for item in json.loads(args.api_corpus.read_text(encoding="utf-8"))}
    existing = {}
    if args.output.exists():
        existing = {item["query"]: item for item in json.loads(args.output.read_text(encoding="utf-8"))}

    for query in TOPIC_QUERIES:
        if existing.get(query, {}).get("extract"):
            continue
        api_item = api_records.get(query, {})
        title = api_item.get("title") or query
        record = {"query": query, "requested_title": title, "source": "bn.wikipedia.org", "license": "CC BY-SA"}
        try:
            final_title, extract = fetch_page(title, args.extract_chars)
            record.update({"title": final_title, "extract": extract})
        except Exception as exc:
            record["error"] = repr(exc)
            # Preserve a clean API extract when direct HTML retrieval fails.
            if api_item.get("extract"):
                record.update({"title": api_item.get("title", title), "extract": api_item["extract"], "fallback": "api"})
        existing[query] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        print(query, "->", record.get("title"), "chars", len(record.get("extract", "")), record.get("error", ""), flush=True)
        time.sleep(args.sleep)

    print("wrote", args.output, "topics", len(existing), "with_extract", sum(bool(x.get("extract")) for x in existing.values()))


if __name__ == "__main__":
    main()
