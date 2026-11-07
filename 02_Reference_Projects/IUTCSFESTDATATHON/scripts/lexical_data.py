from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from datathon_utils import OUTPUT_DIR, clean_text


BAGDHARA_DATASET = "sakhadib/bagdhara-bangla-idioms-dataset"
BENGALI_DICTIONARY_REPO = "https://github.com/MinhasKamal/BengaliDictionary.git"


def ensure_bagdhara(cache_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required for Bagdhara download; install requirements.txt") from exc
    path = Path(kagglehub.dataset_download(BAGDHARA_DATASET))
    return path


def ensure_git_repo(repo_url: str, target: Path) -> Path | None:
    if target.exists():
        return target
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
        return target
    except Exception as exc:
        print("WARN could not clone", repo_url, repr(exc))
        return None


def list_text(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "; ".join(clean_text(x) for x in value if clean_text(x))
    return clean_text(value)


def bagdhara_context(item: dict) -> str:
    parts = [
        f"বাগধারা: {clean_text(item.get('idiom'))}",
        f"শাব্দিক অর্থ: {clean_text(item.get('literal_meaning'))}",
        f"ভাবার্থ: {clean_text(item.get('figurative_meaning_bn'))}",
    ]
    alt = list_text(item.get("alternative_idioms"))
    if alt:
        parts.append(f"বিকল্প রূপ: {alt}")
    examples = list_text(item.get("example_sentences_in_bangla"))
    if examples:
        parts.append(f"উদাহরণ: {examples}")
    domains = list_text(item.get("usage_domain"))
    if domains:
        parts.append(f"ব্যবহারক্ষেত্র: {domains}")
    note = clean_text(item.get("note"))
    if note:
        parts.append(f"নোট: {note}")
    return " ".join(parts)


def add_bagdhara_rows(rows: list[dict], item: dict, source_file: str) -> None:
    idiom = clean_text(item.get("idiom"))
    literal = clean_text(item.get("literal_meaning"))
    figurative = clean_text(item.get("figurative_meaning_bn"))
    if not idiom:
        return
    forms = [idiom] + [clean_text(x) for x in item.get("alternative_idioms") or [] if clean_text(x)]
    context = bagdhara_context(item)
    for form in dict.fromkeys(forms):
        if figurative:
            rows.append(
                {
                    "question": f'"{form}" এর ভাবার্থ কী?',
                    "answer": figurative,
                    "context": context,
                    "source_repo": f"kaggle:{BAGDHARA_DATASET}",
                    "source_file": source_file,
                    "license": "kaggle-public-unknown",
                }
            )
        if literal:
            rows.append(
                {
                    "question": f'"{form}" এর শাব্দিক অর্থ কী?',
                    "answer": literal,
                    "context": context,
                    "source_repo": f"kaggle:{BAGDHARA_DATASET}",
                    "source_file": source_file,
                    "license": "kaggle-public-unknown",
                }
            )


def load_bagdhara(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(root.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        item = json.loads(path.read_text(encoding="utf-8-sig"))
        add_bagdhara_rows(rows, item, path.name)
    return pd.DataFrame(rows)


def load_bengali_dictionary(repo_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    json_path = repo_dir / "BengaliDictionary.json"
    if json_path.exists():
        items = json.loads(json_path.read_text(encoding="utf-8-sig"))
        for item in items:
            bn = clean_text(item.get("bn"))
            en = clean_text(item.get("en"))
            synonyms = [clean_text(x) for x in item.get("bn_syns") or [] if clean_text(x)]
            if not bn or not en:
                continue
            answer = "; ".join(dict.fromkeys([bn] + synonyms[:8]))
            context = f"English word: {en}. বাংলা অর্থ: {answer}."
            rows.append(
                {
                    "question": f'"{en}" এর বাংলা অর্থ কী?',
                    "answer": answer,
                    "context": context,
                    "source_repo": "github:MinhasKamal/BengaliDictionary",
                    "source_file": "BengaliDictionary.json",
                    "license": "GPL-3.0",
                }
            )
    for file_name in ["BengaliDictionary_36.csv", "BengaliDictionary_93.csv"]:
        path = repo_dir / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", header=None, usecols=[0, 1]).fillna("")
        for en, bn in df.itertuples(index=False):
            en_text = clean_text(en)
            bn_text = clean_text(bn)
            if not en_text or not bn_text:
                continue
            rows.append(
                {
                    "question": f'"{en_text}" এর বাংলা অর্থ কী?',
                    "answer": bn_text,
                    "context": f"English word: {en_text}. বাংলা অর্থ: {bn_text}.",
                    "source_repo": "github:MinhasKamal/BengaliDictionary",
                    "source_file": file_name,
                    "license": "GPL-3.0",
                }
            )
    return pd.DataFrame(rows)


def write_corpus(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = df.fillna("").drop_duplicates(["question", "answer", "context"]).reset_index(drop=True)
    df.to_json(output.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    df.to_csv(output.with_suffix(".csv"), index=False)
    print("wrote", output.with_suffix(".jsonl"), output.with_suffix(".csv"), "rows", len(df))
    print(df["source_repo"].value_counts().to_string())
    print(df["license"].value_counts().to_string())


def merge_corpora(paths: list[Path], output: Path) -> None:
    frames = []
    for path in paths:
        if not path.exists():
            print("WARN missing corpus", path)
            continue
        frames.append(pd.read_json(path, lines=True))
    if not frames:
        raise ValueError("no corpora to merge")
    write_corpus(pd.concat(frames, ignore_index=True), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default=str(OUTPUT_DIR / "lexical_evidence_corpus"))
    parser.add_argument("--cache-dir", default=str(OUTPUT_DIR / "source_cache"))
    parser.add_argument("--skip-bagdhara", action="store_true")
    parser.add_argument("--skip-bengali-dictionary", action="store_true")
    parser.add_argument("--merge-with", action="append", default=[])
    parser.add_argument("--merged-output-prefix", default=str(OUTPUT_DIR / "merged_evidence_corpus"))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    if not args.skip_bagdhara:
        bagdhara_dir = ensure_bagdhara(cache_dir)
        frames.append(load_bagdhara(bagdhara_dir))

    if not args.skip_bengali_dictionary:
        repo_dir = ensure_git_repo(BENGALI_DICTIONARY_REPO, cache_dir / "BengaliDictionary")
        if repo_dir is not None:
            frames.append(load_bengali_dictionary(repo_dir))

    if not frames:
        raise ValueError("no lexical sources were loaded")
    output = Path(args.output_prefix)
    lexical = pd.concat(frames, ignore_index=True)
    write_corpus(lexical, output)

    merge_paths = [Path(x) for x in args.merge_with]
    if merge_paths:
        merge_corpora(merge_paths + [output.with_suffix(".jsonl")], Path(args.merged_output_prefix))


if __name__ == "__main__":
    main()
