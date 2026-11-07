from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR


STABLE_FACTS = [
    ("বাংলাদেশের রাজধানী কী?", "ঢাকা", ["চট্টগ্রাম", "খুলনা", "কলকাতা"]),
    ("ভারতের রাজধানী কী?", "নয়াদিল্লি", ["মুম্বাই", "কলকাতা", "চেন্নাই"]),
    ("জাপানের রাজধানী কী?", "টোকিও", ["ওসাকা", "কিয়োটো", "বেইজিং"]),
    ("ফ্রান্সের রাজধানী কী?", "প্যারিস", ["লন্ডন", "বার্লিন", "মাদ্রিদ"]),
    ("পৃথিবীর বৃহত্তম মহাসাগর কোনটি?", "প্রশান্ত মহাসাগর", ["আটলান্টিক মহাসাগর", "ভারত মহাসাগর", "আর্কটিক মহাসাগর"]),
    ("মানবদেহে রক্ত পাম্প করে কোন অঙ্গ?", "হৃদপিণ্ড", ["ফুসফুস", "যকৃত", "কিডনি"]),
    ("পানির রাসায়নিক সংকেত কী?", "H2O", ["CO2", "O2", "NaCl"]),
    ("সূর্যের নিকটতম গ্রহ কোনটি?", "বুধ", ["শুক্র", "মঙ্গল", "বৃহস্পতি"]),
    ("বাংলাদেশের স্বাধীনতা দিবস কবে?", "২৬ মার্চ", ["১৬ ডিসেম্বর", "২১ ফেব্রুয়ারি", "১৫ আগস্ট"]),
    ("বাংলাদেশের বিজয় দিবস কবে?", "১৬ ডিসেম্বর", ["২৬ মার্চ", "২১ ফেব্রুয়ারি", "৭ মার্চ"]),
    ("আন্তর্জাতিক মাতৃভাষা দিবস কবে?", "২১ ফেব্রুয়ারি", ["২৬ মার্চ", "১৬ ডিসেম্বর", "১ মে"]),
    ("জাতীয় কবি কাজী নজরুল ইসলাম কোথায় জন্মগ্রহণ করেন?", "চুরুলিয়া", ["জোড়াসাঁকো", "টুঙ্গিপাড়া", "কাঁঠালপাড়া"]),
    ("রবীন্দ্রনাথ ঠাকুর কোন গ্রন্থের জন্য নোবেল পুরস্কার পান?", "গীতাঞ্জলি", ["সোনার তরী", "বলাকা", "শেষের কবিতা"]),
    ("‘বিদ্রোহী’ কবিতার রচয়িতা কে?", "কাজী নজরুল ইসলাম", ["রবীন্দ্রনাথ ঠাকুর", "জীবনানন্দ দাশ", "সুকান্ত ভট্টাচার্য"]),
    ("‘আমার সোনার বাংলা’ গানের রচয়িতা কে?", "রবীন্দ্রনাথ ঠাকুর", ["কাজী নজরুল ইসলাম", "লালন শাহ", "দ্বিজেন্দ্রলাল রায়"]),
    ("‘অসমাপ্ত আত্মজীবনী’ কার লেখা?", "শেখ মুজিবুর রহমান", ["জিয়াউর রহমান", "মওলানা ভাসানী", "তাজউদ্দীন আহমদ"]),
]


IDIOMS = [
    ("\"ধান্ধা\" এর ভাবার্থ কী?", "অসৎ উদ্দেশ্য বা কৌশল", ["ধীরে ধীরে খাওয়া", "অত্যন্ত আনন্দ", "চুপ থাকা"]),
    ("\"নিজের ঢাক নিজে পেটা\" এর ভাবার্থ কী?", "নিজের প্রশংসা নিজে করা", ["অন্যকে সাহায্য করা", "গোপন রাখা", "ভয় পাওয়া"]),
    ("\"আহ্লাদে আটখানা\" এর ভাবার্থ কী?", "অত্যন্ত আনন্দিত হওয়া", ["অত্যন্ত রাগ করা", "মিথ্যা বলা", "চুপ থাকা"]),
    ("\"চোখের মণি\" এর ভাবার্থ কী?", "অত্যন্ত প্রিয় ব্যক্তি", ["অপ্রিয় ব্যক্তি", "অন্ধকার", "চোখের রোগ"]),
    ("\"দুধের মাছি\" এর ভাবার্থ কী?", "সুযোগসন্ধানী ব্যক্তি", ["সাহসী ব্যক্তি", "পরিশ্রমী ব্যক্তি", "বোকা লোক"]),
    ("\"কাঁটা ঘায়ে নুনের ছিটা\" এর ভাবার্থ কী?", "দুঃখের উপর আরও দুঃখ দেওয়া", ["দুঃখ দূর করা", "অতিথি আপ্যায়ন", "খুশি হওয়া"]),
    ("\"আকাশ কুসুম\" এর ভাবার্থ কী?", "অবাস্তব কল্পনা", ["সহজ কাজ", "বাস্তব ঘটনা", "অল্প লাভ"]),
    ("\"চোখে ধুলো দেওয়া\" এর ভাবার্থ কী?", "প্রতারণা করা", ["সত্য বলা", "সম্মান করা", "সহযোগিতা করা"]),
    ("\"হাতেনাতে ধরা\" এর ভাবার্থ কী?", "অপরাধ করার সময় ধরা পড়া", ["হাত ধোয়া", "সহায়তা করা", "লুকিয়ে থাকা"]),
    ("\"গাছে কাঁঠাল গোঁফে তেল\" এর ভাবার্থ কী?", "কাজের আগেই ফল ভোগের আশা", ["ফল পাকানো", "পরিশ্রম করা", "খুশি হওয়া"]),
]


def add_fact_rows(rows: list[dict], rng: random.Random) -> None:
    for prompt, answer, wrongs in STABLE_FACTS:
        rows.append({"context": "", "prompt_bn": prompt, "response_bn": answer, "label": 1, "source": "curated_fact"})
        for wrong in wrongs:
            rows.append({"context": "", "prompt_bn": prompt, "response_bn": wrong, "label": 0, "source": "curated_fact_wrong"})
        rows.append(
            {
                "context": "",
                "prompt_bn": prompt,
                "response_bn": f"{answer} নয়, সঠিক উত্তর {rng.choice(wrongs)}",
                "label": 0,
                "source": "curated_fact_contradiction",
            }
        )


def add_idiom_rows(rows: list[dict]) -> None:
    for prompt, answer, wrongs in IDIOMS:
        rows.append({"context": "", "prompt_bn": prompt, "response_bn": answer, "label": 1, "source": "curated_idiom"})
        for wrong in wrongs:
            rows.append({"context": "", "prompt_bn": prompt, "response_bn": wrong, "label": 0, "source": "curated_idiom_wrong"})


def add_math_rows(rows: list[dict]) -> None:
    for a in range(2, 21):
        for b in range(2, 11):
            if len(rows) > 600:
                break
            correct = a + b
            prompt = f"{a} এবং {b} এর যোগফল কত?"
            rows.append({"context": "", "prompt_bn": prompt, "response_bn": str(correct), "label": 1, "source": "curated_math"})
            rows.append({"context": "", "prompt_bn": prompt, "response_bn": str(correct + 1), "label": 0, "source": "curated_math_wrong"})
            prompt2 = f"{a*b} কে {b} দিয়ে ভাগ করলে ভাগফল কত?"
            rows.append({"context": "", "prompt_bn": prompt2, "response_bn": str(a), "label": 1, "source": "curated_math"})
            rows.append({"context": "", "prompt_bn": prompt2, "response_bn": str(a + 2), "label": 0, "source": "curated_math_wrong"})


def add_context_rows(rows: list[dict]) -> None:
    for prompt, answer, wrongs in STABLE_FACTS[:12]:
        context = f"তথ্যসূত্র অনুযায়ী, {prompt.replace(' কী?', '')} এর উত্তর হলো {answer}। এই তথ্যটি যাচাই করা হয়েছে।"
        rows.append({"context": context, "prompt_bn": prompt, "response_bn": answer, "label": 1, "source": "curated_context"})
        rows.append({"context": context, "prompt_bn": prompt, "response_bn": wrongs[0], "label": 0, "source": "curated_context_wrong"})
    for prompt, answer, wrongs in IDIOMS:
        context = f"বাংলা বাগধারা অভিধানে বলা হয়েছে: {prompt.replace(' এর ভাবার্থ কী?', '')} বলতে {answer} বোঝায়।"
        rows.append({"context": context, "prompt_bn": prompt, "response_bn": answer, "label": 1, "source": "curated_context_idiom"})
        rows.append({"context": context, "prompt_bn": prompt, "response_bn": wrongs[0], "label": 0, "source": "curated_context_idiom_wrong"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--jsonl", default=str(OUTPUT_DIR / "curated_synthetic.jsonl"))
    parser.add_argument("--csv", default=str(OUTPUT_DIR / "curated_synthetic.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    add_fact_rows(rows, rng)
    add_idiom_rows(rows)
    add_math_rows(rows)
    add_context_rows(rows)
    rng.shuffle(rows)

    jsonl = Path(args.jsonl)
    csv = Path(args.csv)
    jsonl.parent.mkdir(exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print("wrote", jsonl, csv, "rows", len(rows), "labels", pd.Series([r["label"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
