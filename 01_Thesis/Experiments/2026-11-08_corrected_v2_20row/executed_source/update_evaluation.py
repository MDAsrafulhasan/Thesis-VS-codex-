#!/usr/bin/env python3
"""Run the five-stage BenHalluEval detector and compare it on a fixed holdout.

The implementation follows the supplied architecture with explicit defaults:
Unicode/Bengali normalization; claim and contrastive-query generation; E5 + BM25
reciprocal-rank fusion; BGE reranking; multilingual NLI and XLM-R token recall;
Qwen first-token log-probabilities; and a five-seed, grouped logistic-regression
ensemble with train-only, context-regime thresholds.

Label contract at the evaluator boundary is the benchmark contract:
0 = faithful (Track A), 1 = hallucinated (Track B). The image's displayed final
labels reverse this convention, so this script maps the internal
"hallucinated" decision to benchmark label 1 and records that mapping.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL = ROOT / "01_Thesis/Data/BenHalluEval/Processed/Splits/track_a_b_first10_each.csv"
DEFAULT_TRAIN = ROOT / "01_Thesis/Data/BenHalluEval/Processed/Splits/benhallu_stratified_10pct.csv"
DEFAULT_BASELINE = ROOT / "01_Thesis/Results/BenHalluEval/track_a_b_first10_each_baseline"
DEFAULT_OUTPUT = ROOT / "01_Thesis/Results/BenHalluEval/update_pipeline_20row"

PRIMARY_COLUMNS = {
    "id", "task", "track", "hallucination_type", "question", "context",
    "candidate_answer", "ground_truth_label",
}
TASKS = {"bn_qa", "codemixed_qa", "summarization", "math"}
LABEL_BY_TRACK = {"Track_A": 0, "Track_B": 1}
HALLUCINATION_LABEL = 1
WEIGHTS = (0.5, 0.5)
SEEDS = (42, 43, 44, 45, 46)
C_VALUES = (0.1, 1.0, 10.0)
N_FOLDS = 5
RRF_K = 60
RERANK_TOP_K = 60
EVIDENCE_TOP_N = 5
MAX_CLAIMS = 3
MAX_LEN = 512

E5_MODEL_DEFAULT = "intfloat/multilingual-e5-large"
RERANKER_MODEL_DEFAULT = "BAAI/bge-reranker-v2-m3"
NLI_MODEL_DEFAULT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
XLMR_MODEL_DEFAULT = "FacebookAI/xlm-roberta-base"
QG_MODEL_DEFAULT = "qwen3:1.7b"
QA_MODEL_DEFAULT = "gemma3:1b"
LOGIT_MODEL_DEFAULT = "qwen3:1.7b"
OLLAMA_DEFAULT = "http://127.0.0.1:11436"

# Conservative common-word map for informal romanized Bengali. Unknown Latin
# words are retained and counted; no guessed phoneme conversion is applied.
# The supplied 20-row evaluation sample is Bengali-script, so this map does not
# alter those evaluation strings. For broader Banglish work, pass a reviewed
# JSON word map through --banglish_map.
DEFAULT_BANGLISH_MAP = {
    "ami": "আমি", "amra": "আমরা", "amar": "আমার", "amader": "আমাদের",
    "tumi": "তুমি", "tomar": "তোমার", "tomra": "তোমরা", "apni": "আপনি",
    "apnar": "আপনার", "ki": "কি", "kno": "কেন", "keno": "কেন",
    "kothay": "কোথায়", "kokhon": "কখন", "kivabe": "কিভাবে",
    "bhalo": "ভালো", "valo": "ভালো", "khub": "খুব", "onek": "অনেক",
    "bangla": "বাংলা", "bangladesh": "বাংলাদেশ", "dhaka": "ঢাকা",
    "ache": "আছে", "ase": "আছে", "nei": "নেই", "nai": "নাই",
    "hoy": "হয়", "hocche": "হচ্ছে", "chilo": "ছিল", "kore": "করে",
    "kora": "করা", "koreche": "করেছে", "hya": "হ্যাঁ", "ha": "হ্যাঁ",
    "na": "না", "ebong": "এবং", "ar": "আর", "theke": "থেকে",
    "jonno": "জন্য", "sathe": "সাথে", "ki?:": "কি?", "doya": "দয়া",
    "kore": "করে", "dhonnobad": "ধন্যবাদ", "shotti": "সত্যি",
    "mithya": "মিথ্যা", "uttor": "উত্তর", "proshno": "প্রশ্ন",
}

ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
ORTHOGRAPHIC_MAP = {"\u09dc": "\u09a1\u09bc", "\u09dd": "\u09a2\u09bc"}
BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
WORD_RE = re.compile(r"[\u0980-\u09ff]+|[A-Za-z]+|\d+(?:[.,]\d+)*")
SENTENCE_RE = re.compile(r"(?<=[।!?])\s+|\n+|\s+[;；]\s+|\s+\u2022\s+")
CLAUSE_RE = re.compile(r"\s+(?:এবং|কিন্তু|তবে|আরও)\s+|\s+(?:and|but|however)\s+", re.I)
NEGATION_WORDS = {"না", "নয়", "নেই", "নাই", "ন" , "not", "never", "no", "without", "false"}

NLI_FEATURES = [
    "whole_entailment_p", "whole_contradiction_p", "whole_neutral_p",
    "claim_entailment_mean", "claim_entailment_min", "claim_contradiction_max",
    "claim_contradiction_mean", "claim_neutral_mean", "claim_margin_mean",
    "claim_margin_min", "claim_support_rate", "claim_contradiction_rate",
    "negated_claim_contradiction_mean", "qa_answer_entailment_p",
    "xlmr_bertscore_recall", "e5_retrieval_margin",
]
LOGIT_FEATURES = ["p_hallucinated", "logit_margin_hall_minus_faith", "verdict_entropy"]
LEXICAL_FEATURES = [
    "token_fertility", "word_jaccard_answer_evidence", "char3_jaccard_answer_evidence",
    "word_precision_answer_evidence", "word_recall_answer_evidence", "word_f1_answer_evidence",
    "numeric_match_ratio", "negation_mismatch", "bge_reranker_margin",
    "question_answer_overlap",
]
FEATURES = NLI_FEATURES + LOGIT_FEATURES + LEXICAL_FEATURES


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonicalize(text: Any, banglish_map: dict[str, str]) -> tuple[str, float, list[str]]:
    """NFKC and Bengali orthographic normalization plus reviewed word mapping."""
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    for source, target in ORTHOGRAPHIC_MAP.items():
        value = value.replace(source, target)
    value = value.translate(BENGALI_DIGITS)
    value = "".join(ch for ch in value if ch not in ZERO_WIDTH)
    words = WORD_RE.findall(value)
    latin_words = [w for w in words if re.fullmatch(r"[A-Za-z]+", w)]
    mapped = {k.casefold(): v for k, v in banglish_map.items()}
    unknown = []
    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        if not re.fullmatch(r"[A-Za-z]+", word):
            return word
        replacement = mapped.get(word.casefold())
        if replacement is None:
            unknown.append(word)
            return word
        return replacement
    value = re.sub(r"[A-Za-z]+", replace_word, value)
    value = re.sub(r"\s+", " ", value).strip()
    mapped_count = len(latin_words) - len(unknown)
    mapped_rate = mapped_count / len(latin_words) if latin_words else 1.0
    return value, float(mapped_rate), unknown


def word_tokens(text: str) -> list[str]:
    return [w.casefold() for w in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]


def fallback_claims(answer: str, question: str) -> list[dict[str, str]]:
    pieces: list[str] = []
    for sentence in split_sentences(answer) or [answer.strip()]:
        pieces.extend(p.strip(" \t,،") for p in CLAUSE_RE.split(sentence) if p.strip(" \t,،"))
    claims = []
    seen = set()
    for piece in pieces:
        key = piece.casefold()
        if not piece or key in seen:
            continue
        seen.add(key)
        claims.append({
            "claim": piece,
            "positive_query": f"{question} দাবিটি যাচাই করুন: {piece}",
            "negative_query": f"{question} দাবিটির বিপরীত সমর্থিত কি: {piece}",
            "negative_claim": f"এটি সত্য নয় যে {piece}",
        })
        if len(claims) >= MAX_CLAIMS:
            break
    return claims or [{
        "claim": answer.strip(),
        "positive_query": f"{question} দাবিটি যাচাই করুন: {answer.strip()}",
        "negative_query": f"{question} দাবিটির বিপরীত সমর্থিত কি: {answer.strip()}",
        "negative_claim": f"এটি সত্য নয় যে {answer.strip()}",
    }]


def safe_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        want_logprobs: bool = False,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "seed": 42, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        if want_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        request = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc

    def model_digests(self, models: Sequence[str]) -> dict[str, str]:
        request = urllib.request.Request(self.base_url + "/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return {model: f"unavailable: {type(exc).__name__}: {exc}" for model in models}
        tags = {str(item.get("name")): str(item.get("digest", "unknown")) for item in payload.get("models", [])}
        result = {}
        for model in models:
            result[model] = tags.get(model, tags.get(model + ":latest", "not found in local /api/tags"))
        return result


def is_repetition_collapse(text: str, completion_tokens: int | None, max_tokens: int, closing: str) -> bool:
    words = [w.casefold() for w in re.findall(r"[\w\u0980-\u09ff]+", text)]
    for n in range(5, len(words) // 4 + 1):
        for start in range(0, len(words) - n * 4 + 1):
            gram = words[start:start + n]
            if all(words[start + repeat * n:start + (repeat + 1) * n] == gram for repeat in range(1, 4)):
                return True
    return bool(completion_tokens is not None and completion_tokens >= max_tokens and closing not in text)


def parse_claims_response(raw: str, question: str, answer: str) -> tuple[list[dict[str, str]], str | None]:
    data = safe_parse_json(raw)
    claims_raw = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims_raw, list):
        return fallback_claims(answer, question), "invalid_qg_json"
    claims = []
    for item in claims_raw[:MAX_CLAIMS]:
        if isinstance(item, str):
            claim = item.strip()
            pos = f"{question} দাবিটি যাচাই করুন: {claim}"
            neg = f"{question} দাবিটির বিপরীত সমর্থিত কি: {claim}"
            negated = f"এটি সত্য নয় যে {claim}"
        elif isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            pos = str(item.get("positive_query", "")).strip() or f"{question} দাবিটি যাচাই করুন: {claim}"
            neg = str(item.get("negative_query", "")).strip() or f"{question} দাবিটির বিপরীত সমর্থিত কি: {claim}"
            negated = str(item.get("negative_claim", "")).strip() or f"এটি সত্য নয় যে {claim}"
        else:
            continue
        if claim:
            claims.append({"claim": claim, "positive_query": pos, "negative_query": neg, "negative_claim": negated})
    if not claims:
        return fallback_claims(answer, question), "empty_qg_claims"
    return claims, None


def qg_prompt(question: str, answer: str) -> str:
    return (
        "Extract up to three short, independently verifiable factual claims from the candidate answer. "
        "Keep each claim concise and do not add facts. Return JSON only in the form "
        '{"claims":["claim 1","claim 2"]}. Do not include queries or commentary.\n'
        f"Question: {question}\nCandidate answer: {answer}"
    )


def logit_prompt(question: str, context: str, answer: str) -> str:
    return (
        "Decide whether the candidate answer contains a factual hallucination. "
        "Use the supplied context when present. If no context is supplied, judge factual consistency conservatively. "
        "Output exactly one token: 1 means hallucinated; 0 means faithful.\n"
        f"Question: {question}\nContext: {context or '[closed book]'}\n"
        f"Candidate answer: {answer}\nLabel:"
    )


def first_token_probabilities(response: dict[str, Any]) -> tuple[float | None, float | None, float | None, str | None]:
    sequence = response.get("logprobs") or []
    if not sequence:
        return None, None, None, "missing_logprobs"
    candidates = sequence[0].get("top_logprobs") or []
    found: dict[str, float] = {}
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip()
        if token in {"0", "1"}:
            found[token] = float(candidate["logprob"])
    if not {"0", "1"}.issubset(found):
        return None, None, None, "binary_token_not_in_top_logprobs"
    l0, l1 = found["0"], found["1"]
    maximum = max(l0, l1)
    e0, e1 = math.exp(l0 - maximum), math.exp(l1 - maximum)
    denom = e0 + e1
    p_faith, p_hall = e0 / denom, e1 / denom
    margin = l1 - l0
    entropy = -(p_faith * math.log(max(p_faith, 1e-15)) + p_hall * math.log(max(p_hall, 1e-15)))
    return p_hall, margin, entropy, None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_hf_revision(model_name: str, cache_dir: Path | None) -> str:
    """Return the exact Hugging Face commit resolved in the local cache."""
    from huggingface_hub.constants import HF_HUB_CACHE
    cache_root = cache_dir or Path(os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE))
    repo_cache = cache_root / ("models--" + model_name.replace("/", "--"))
    ref = repo_cache / "refs" / "main"
    if ref.exists():
        commit = ref.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit):
            return commit
    snapshots = repo_cache / "snapshots"
    candidates = sorted(p.name for p in snapshots.iterdir() if p.is_dir()) if snapshots.exists() else []
    if len(candidates) == 1 and re.fullmatch(r"[0-9a-f]{40}", candidates[0]):
        return candidates[0]
    raise RuntimeError(f"Could not resolve a pinned Hugging Face revision for {model_name} in {repo_cache}")


def read_primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    missing = PRIMARY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing primary columns: {sorted(missing)}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["track"] = frame["track"].astype(str)
    frame["task"] = frame["task"].astype(str)
    frame["ground_truth_label"] = pd.to_numeric(frame["ground_truth_label"], errors="raise").astype(int)
    if frame["id"].str.strip().eq("").any() or frame["id"].duplicated().any():
        raise ValueError(f"{path} has empty or duplicate IDs")
    if not set(frame["track"]).issubset(LABEL_BY_TRACK):
        raise ValueError(f"{path} has unknown track values")
    expected = frame["track"].map(LABEL_BY_TRACK).astype(int)
    if not frame["ground_truth_label"].equals(expected):
        raise ValueError(f"{path} does not use benchmark labels 0=Track_A, 1=Track_B")
    if not set(frame["task"]).issubset(TASKS):
        raise ValueError(f"{path} contains unknown tasks")
    return frame


def load_banglish_map(path: Path | None) -> dict[str, str]:
    mapping = dict(DEFAULT_BANGLISH_MAP)
    if path is not None:
        user_map = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(user_map, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in user_map.items()):
            raise ValueError("--banglish_map must be a UTF-8 JSON object mapping romanized words to Bengali strings")
        mapping.update(user_map)
    return mapping


def prepare_data(train_path: Path, eval_path: Path, banglish_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    train = read_primary(train_path)
    evaluation = read_primary(eval_path)
    if evaluation.empty or not {"Track_A", "Track_B"}.issubset(set(evaluation.track)):
        raise ValueError("Evaluation CSV must contain rows from both Track_A and Track_B")
    for frame in (train, evaluation):
        normalized, mapping_rates, unknowns = [], [], []
        for _, row in frame.iterrows():
            q, qrate, qunknown = canonicalize(row.question, banglish_map)
            a, arate, aunknown = canonicalize(row.candidate_answer, banglish_map)
            c, crate, cunknown = canonicalize(row.context, banglish_map)
            normalized.append((q, a, c))
            rates = [r for r in (qrate, arate, crate) if r < 1.0]
            mapping_rates.append(float(np.mean(rates)) if rates else 1.0)
            unknowns.append(sorted(set(qunknown + aunknown + cunknown)))
        frame["norm_question"] = [x[0] for x in normalized]
        frame["norm_answer"] = [x[1] for x in normalized]
        frame["norm_context"] = [x[2] for x in normalized]
        frame["group_key"] = frame["norm_question"].map(lambda x: sha256_text(x))
        frame["roman_map_rate"] = mapping_rates
        frame["roman_oov_tokens"] = unknowns
    eval_groups = set(evaluation.group_key)
    before = len(train)
    train = train.loc[~train.group_key.isin(eval_groups)].copy().reset_index(drop=True)
    dropped = before - len(train)
    remaining_id_overlap = set(train.id) & set(evaluation.id)
    remaining_group_overlap = set(train.group_key) & set(evaluation.group_key)
    if remaining_id_overlap or remaining_group_overlap:
        raise RuntimeError("Leakage guard failed: evaluation IDs or question groups remain in training data")
    return train, evaluation.reset_index(drop=True), dropped, len(train)


def row_fingerprint(row: dict[str, Any], split: str) -> str:
    fields = [split, str(row["id"]), str(row["norm_question"]), str(row["norm_context"]), str(row["norm_answer"])]
    return sha256_text("\n".join(fields))


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(row, dict) and row.get("id"):
                latest[str(row["id"])] = row
    return latest


def append_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def sentence_docs(text: str, row_id: str, group_key: str, source: str) -> list[dict[str, str]]:
    return [
        {"doc_id": f"{row_id}:sent:{i}", "text": sentence, "group_key": group_key, "source": source}
        for i, sentence in enumerate(split_sentences(text)) if sentence
    ]


def tokenize_for_bm25(text: str) -> list[str]:
    return [token for token in word_tokens(text) if token.strip()]


class BM25Index:
    """Small in-file BM25Okapi implementation, k1=1.5 and b=0.75."""
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.tokens = [tokenize_for_bm25(d) for d in self.documents]
        self.k1, self.b = k1, b
        self.avgdl = float(np.mean([len(t) for t in self.tokens])) if self.tokens else 0.0
        df = Counter(token for doc in self.tokens for token in set(doc))
        n = len(self.tokens)
        self.idf = {term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.term_freq = [Counter(doc) for doc in self.tokens]

    def scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.zeros(0, dtype=float)
        qtokens = set(tokenize_for_bm25(query))
        scores = np.zeros(len(self.documents), dtype=float)
        for i, tf in enumerate(self.term_freq):
            dl = len(self.tokens[i])
            norm = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-12))
            for token in qtokens:
                f = tf.get(token, 0)
                if f:
                    scores[i] += self.idf.get(token, 0.0) * f * (self.k1 + 1.0) / (f + norm)
        return scores


def batched(iterable: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start:start + batch_size]


def build_evidence_corpus(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[list[dict[str, str]], set[str]]:
    eval_answer_hashes = {sha256_text(str(v)) for v in evaluation.norm_answer}
    docs: list[dict[str, str]] = []
    seen = set()
    for _, row in train.iterrows():
        for doc in sentence_docs(row.norm_context, str(row.id), str(row.group_key), "train_context"):
            digest = sha256_text(doc["text"])
            if digest in eval_answer_hashes or digest in seen:
                continue
            seen.add(digest)
            doc["sha256"] = digest
            docs.append(doc)
    return docs, eval_answer_hashes


def build_queries(cache_record: dict[str, Any]) -> list[str]:
    claims = cache_record.get("claims") or []
    queries = []
    for claim in claims:
        queries.extend([claim["positive_query"], claim["negative_query"]])
    if not queries:
        queries.append(str(cache_record.get("norm_question", "")))
    return list(dict.fromkeys(q for q in queries if q))


def stage1_qg_and_logits(
    rows: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    ollama: OllamaClient,
    qg_model: str,
    logit_model: str,
) -> None:
    total = len(rows)
    for index, (_, row_s) in enumerate(rows.iterrows(), start=1):
        row = row_s.to_dict()
        split = str(row["split"])
        item_id = str(row["id"])
        fingerprint = row_fingerprint(row, split)
        record = cache.get(item_id, {"id": item_id})
        if record.get("input_fingerprint") != fingerprint:
            record = {
                "id": item_id, "input_fingerprint": fingerprint, "split": split,
                "track": str(row["track"]), "task": str(row["task"]),
                "hallucination_type": str(row["hallucination_type"]),
                "ground_truth_label": int(row["ground_truth_label"]),
                "group_key": str(row["group_key"]),
                "norm_question": str(row["norm_question"]),
                "norm_context": str(row["norm_context"]),
                "norm_answer": str(row["norm_answer"]),
                "raw_question": str(row["question"]), "raw_context": str(row["context"]),
                "raw_answer": str(row["candidate_answer"]),
                "roman_map_rate": float(row["roman_map_rate"]),
                "roman_oov_tokens": row["roman_oov_tokens"],
            }
        if "claims" not in record:
            raw_qg = ""
            prompt = qg_prompt(record["norm_question"], record["norm_answer"])
            qg_error = None
            qg_response: dict[str, Any] = {}
            try:
                qg_response = ollama.generate(qg_model, prompt, max_tokens=256, json_mode=True)
                raw_qg = str(qg_response.get("response", ""))
                claims, qg_error = parse_claims_response(raw_qg, record["norm_question"], record["norm_answer"])
            except Exception as exc:
                claims, qg_error = fallback_claims(record["norm_answer"], record["norm_question"]), f"{type(exc).__name__}: {exc}"
            record.update({
                "claims": claims, "qg_raw_prompt": prompt, "qg_raw_response": raw_qg,
                "qg_error": qg_error, "qg_fallback": qg_error is not None,
                "qg_completion_tokens": qg_response.get("eval_count"),
                "qg_collapsed": is_repetition_collapse(raw_qg, qg_response.get("eval_count"), 256, "}"),
            })
            cache[item_id] = record
            append_cache(cache_path, record)
        if "llm_logit_features" not in record:
            raw_logit = ""
            prompt = logit_prompt(record["norm_question"], record["norm_context"], record["norm_answer"])
            logit_response: dict[str, Any] = {}
            try:
                logit_response = ollama.generate(
                    logit_model,
                    prompt,
                    max_tokens=1,
                    want_logprobs=True,
                )
                raw_logit = str(logit_response.get("response", ""))
                p_hall, margin, entropy, logit_error = first_token_probabilities(logit_response)
            except Exception as exc:
                p_hall, margin, entropy = None, None, None
                logit_error = f"{type(exc).__name__}: {exc}"
            record.update({
                "llm_logit_features": {"p_hallucinated": p_hall, "logit_margin_hall_minus_faith": margin, "verdict_entropy": entropy},
                "llm_logit_raw_prompt": prompt, "llm_logit_raw_response": raw_logit, "llm_logit_error": logit_error,
                "llm_logit_completion_tokens": logit_response.get("eval_count"),
                "llm_logit_collapsed": False,
            })
            cache[item_id] = record
            append_cache(cache_path, record)
        if index % 25 == 0 or index == total:
            print(f"Stage 1/4 QG + first-token logits: {index}/{total}", flush=True)


def stage2_retrieve(
    rows: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    e5_model_name: str,
    reranker_name: str,
    cache_dir: Path | None,
    device: str,
) -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoTokenizer

    needed = [str(r.id) for _, r in rows.iterrows() if "evidence" not in cache.get(str(r.id), {})]
    if not needed:
        print("Stage 2 retrieval: cached for all rows", flush=True)
        return
    corpus, eval_hashes = build_evidence_corpus(train, evaluation)
    global_texts = list(dict.fromkeys(d["text"] for d in corpus))
    tokenizer = AutoTokenizer.from_pretrained(e5_model_name, cache_dir=str(cache_dir) if cache_dir else None)
    e5 = SentenceTransformer(e5_model_name, device=device, cache_folder=str(cache_dir) if cache_dir else None)
    global_vectors = e5.encode(
        ["passage: " + text for text in global_texts],
        batch_size=8, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    ) if global_texts else np.zeros((0, 1), dtype=np.float32)
    vector_by_text = {text: global_vectors[i] for i, text in enumerate(global_texts)}
    global_bm25 = BM25Index(global_texts)
    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    for index, (_, row_s) in enumerate(rows.iterrows(), start=1):
        row = row_s.to_dict()
        item_id = str(row["id"])
        record = cache[item_id]
        if "evidence_candidates" in record:
            candidates_by_id[item_id] = record["evidence_candidates"]
            continue
        record["e5_token_fertility"] = _token_fertility(tokenizer, record["norm_answer"])
        if str(row["norm_context"]).strip():
            docs = sentence_docs(str(row["norm_context"]), item_id, str(row["group_key"]), "row_context")
            local_texts = list(dict.fromkeys(d["text"] for d in docs))
            missing_texts = [t for t in local_texts if t not in vector_by_text]
            if missing_texts:
                missing_vectors = e5.encode(
                    ["passage: " + t for t in missing_texts],
                    convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
                )
                vector_by_text.update({t: missing_vectors[i] for i, t in enumerate(missing_texts)})
            local_by_text = {t: vector_by_text[t] for t in local_texts}
            bm25 = BM25Index(local_texts)
            source = "row_context"
        else:
            docs = [d for d in corpus if d["group_key"] != str(row["group_key"]) and d.get("sha256") not in eval_hashes]
            local_texts = list(dict.fromkeys(d["text"] for d in docs))
            local_by_text = vector_by_text
            bm25 = BM25Index(local_texts)
            source = "training_context_corpus"
        if not local_texts:
            record.update({"evidence_candidates": [], "retrieval_source": source, "e5_retrieval_margin": 0.0, "decontamination_hashes": len(eval_hashes)})
            candidates_by_id[item_id] = []
            cache[item_id] = record
            append_cache(cache_path, record)
            continue
        queries = build_queries(record)
        q_vectors = e5.encode(["query: " + q for q in queries], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        doc_text_by_id = {d["doc_id"]: d["text"] for d in docs}
        doc_ids_by_text: defaultdict[str, list[str]] = defaultdict(list)
        for d in docs:
            doc_ids_by_text[d["text"]].append(d["doc_id"])
        for qi, query in enumerate(queries):
            lexical_scores = bm25.scores(query)
            lexical_order = np.argsort(-lexical_scores)[:RERANK_TOP_K]
            query_vector = q_vectors[qi]
            dense_scores = np.asarray([float(np.dot(query_vector, local_by_text[t])) for t in local_texts])
            dense_order = np.argsort(-dense_scores)[:RERANK_TOP_K]
            for rank, pos in enumerate(lexical_order, start=1):
                for doc_id in doc_ids_by_text.get(local_texts[int(pos)], []):
                    rrf_scores[doc_id] += 1.0 / (RRF_K + rank)
            for rank, pos in enumerate(dense_order, start=1):
                for doc_id in doc_ids_by_text.get(local_texts[int(pos)], []):
                    rrf_scores[doc_id] += 1.0 / (RRF_K + rank)
        ranked = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:RERANK_TOP_K]
        candidates = [{"doc_id": doc_id, "text": doc_text_by_id[doc_id], "rrf_score": float(score)} for doc_id, score in ranked]
        candidate_texts = [d["text"] for d in candidates]
        rrf_vals = sorted((d["rrf_score"] for d in candidates), reverse=True)
        margin = (rrf_vals[0] - rrf_vals[1]) if len(rrf_vals) > 1 else (rrf_vals[0] if rrf_vals else 0.0)
        record.update({
            "evidence_candidates": candidates, "retrieval_source": source,
            "e5_retrieval_margin": float(margin), "decontamination_hashes": len(eval_hashes),
        })
        cache[item_id] = record
        append_cache(cache_path, record)
        candidates_by_id[item_id] = candidates
        if index % 50 == 0 or index == len(rows):
            print(f"Stage 2/4 dense + BM25 retrieval: {index}/{len(rows)}", flush=True)
    del e5, tokenizer, global_vectors, vector_by_text
    gc.collect()
    reranker = CrossEncoder(reranker_name, device=device, max_length=MAX_LEN, cache_folder=str(cache_dir) if cache_dir else None)
    for index, (_, row_s) in enumerate(rows.iterrows(), start=1):
        row = row_s.to_dict()
        item_id = str(row["id"])
        record = cache[item_id]
        if "evidence" in record:
            continue
        candidates = record.get("evidence_candidates", [])[:RERANK_TOP_K]
        if candidates:
            query = record["norm_question"] + " Claim: " + " ; ".join(c["claim"] for c in record.get("claims", []))
            scores = reranker.predict([(query, c["text"]) for c in candidates], batch_size=8, show_progress_bar=False)
            ranked = sorted(
                [{**candidate, "reranker_score": float(score)} for candidate, score in zip(candidates, scores)],
                key=lambda x: (-x["reranker_score"], -x["rrf_score"], x["doc_id"]),
            )
        else:
            ranked = []
        selected = ranked[:EVIDENCE_TOP_N]
        evidence = " ".join(item["text"] for item in selected)
        bge_scores = sorted((item["reranker_score"] for item in ranked), reverse=True)
        bge_margin = (bge_scores[0] - bge_scores[1]) if len(bge_scores) > 1 else (bge_scores[0] if bge_scores else 0.0)
        record.update({"evidence": evidence, "reranked_evidence": selected, "bge_reranker_margin": float(bge_margin)})
        cache[item_id] = record
        append_cache(cache_path, record)
        if index % 50 == 0 or index == len(rows):
            print(f"Stage 2/4 BGE reranking: {index}/{len(rows)}", flush=True)
    del reranker
    gc.collect()


def _token_fertility(tokenizer: Any, text: str) -> float:
    words = word_tokens(text)
    if not words:
        return 0.0
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return float(len(encoded) / len(words))


def stage3_qa(
    rows: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    ollama: OllamaClient,
    qa_model: str,
) -> None:
    for index, (_, row_s) in enumerate(rows.iterrows(), start=1):
        item_id = str(row_s.id)
        record = cache[item_id]
        if "qa_answers" in record:
            continue
        evidence = record.get("evidence", "")
        queries = []
        for claim_index, claim in enumerate(record.get("claims", [])):
            queries.append({"query_id": f"c{claim_index}_positive", "query": claim["positive_query"]})
            queries.append({"query_id": f"c{claim_index}_negative", "query": claim["negative_query"]})
        if not queries:
            queries = [{"query_id": "original", "query": record["norm_question"]}]
        prompt = (
            "Answer each question using only the evidence. If the evidence does not answer it, return an empty answer. "
            "Do not judge whether the candidate is correct. Return JSON only: {\"answers\":[{\"query_id\":\"...\",\"answer\":\"...\"}]}.\n"
            f"Evidence: {evidence or '[no retrieved evidence]'}\nQueries: {json.dumps(queries, ensure_ascii=False)}"
        )
        raw = ""
        error = None
        answers: dict[str, str] = {}
        response: dict[str, Any] = {}
        try:
            response = ollama.generate(qa_model, prompt, max_tokens=256, json_mode=True)
            raw = str(response.get("response", ""))
            parsed = safe_parse_json(raw)
            arr = parsed.get("answers") if isinstance(parsed, dict) else None
            if not isinstance(arr, list):
                error = "invalid_qa_json"
            else:
                for item in arr:
                    if isinstance(item, dict) and item.get("query_id") is not None:
                        answers[str(item["query_id"])] = str(item.get("answer", "")).strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        record.update({
            "qa_answers": answers, "qa_raw_prompt": prompt, "qa_raw_response": raw,
            "qa_error": error,
            "qa_completion_tokens": response.get("eval_count"),
            "qa_collapsed": is_repetition_collapse(raw, response.get("eval_count"), 256, "}"),
        })
        cache[item_id] = record
        append_cache(cache_path, record)
        if index % 25 == 0 or index == len(rows):
            print(f"Stage 3/4 decoupled QA: {index}/{len(rows)}", flush=True)


def nli_label_indices(config: Any) -> tuple[int, int, int]:
    labels = {int(k): str(v).casefold() for k, v in config.id2label.items()}
    def find(name: str, fallback: int) -> int:
        for index, label in labels.items():
            if name in label:
                return index
        return fallback
    contra = find("contradiction", 0)
    neutral = find("neutral", 1)
    entail = find("entail", 2)
    if len({contra, neutral, entail}) != 3:
        raise ValueError(f"Could not map NLI output labels from config.id2label={labels}")
    return entail, contra, neutral


def stage3_nli(
    rows: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    model_name: str,
    cache_dir: Path | None,
    device: str,
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    pending: list[tuple[str, str, str, str]] = []
    for _, row in rows.iterrows():
        item_id = str(row.id)
        record = cache[item_id]
        if "nli_feature_scores" in record:
            continue
        evidence = str(record.get("evidence", "")) or str(record.get("norm_question", ""))
        pending.append((item_id, "whole", evidence, str(record["norm_answer"])))
        claims = record.get("claims", [])
        for j, claim in enumerate(claims):
            pending.append((item_id, f"claim_{j}", evidence, claim["claim"]))
            pending.append((item_id, f"negative_{j}", evidence, claim["negative_claim"]))
        qa_answers = record.get("qa_answers", {})
        qa_answer = next((v for k, v in qa_answers.items() if k.endswith("_positive") and v), "")
        record["selected_qa_answer"] = qa_answer
        if qa_answer:
            pending.append((item_id, "qa_answer", evidence, qa_answer))
    if not pending:
        print("Stage 3/4 NLI: cached for all rows", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None).to(device).eval()
    entail_id, contra_id, neutral_id = nli_label_indices(model.config)
    out: defaultdict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    batch_size = 8 if device == "cpu" else 32
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        inputs = tokenizer(
            [p[2] for p in batch], [p[3] for p in batch],
            padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits.float(), dim=-1).cpu().numpy()
        for item, values in zip(batch, probs):
            item_id, key, _, _ = item
            out[item_id][key] = {
                "entailment": float(values[entail_id]),
                "contradiction": float(values[contra_id]),
                "neutral": float(values[neutral_id]),
            }
        if (start // batch_size + 1) % 100 == 0 or start + batch_size >= len(pending):
            print(f"Stage 3/4 mDeBERTa NLI pairs: {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
    for _, row in rows.iterrows():
        item_id = str(row.id)
        record = cache[item_id]
        record["nli_feature_scores"] = out.get(item_id, {})
        cache[item_id] = record
        append_cache(cache_path, record)
    del model, tokenizer
    gc.collect()


def stage3_xlmr_recall(
    rows: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    model_name: str,
    cache_dir: Path | None,
    device: str,
) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    pending = [str(row.id) for _, row in rows.iterrows() if "xlmr_bertscore_recall" not in cache[str(row.id)]]
    if not pending:
        print("Stage 3/4 XLM-R recall: cached for all rows", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None).to(device).eval()
    for index, item_id in enumerate(pending, start=1):
        record = cache[item_id]
        reference = str(record.get("evidence", "")) or str(record.get("norm_question", ""))
        candidate = str(record.get("norm_answer", ""))
        if not reference.strip() or not candidate.strip():
            score = 0.0
        else:
            encoded = tokenizer(
                [reference, candidate], padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state.float()
            ref_mask = encoded["attention_mask"][0].bool()
            cand_mask = encoded["attention_mask"][1].bool()
            special = set(tokenizer.all_special_ids)
            ref_mask &= torch.tensor([int(t) not in special for t in encoded["input_ids"][0]], device=device)
            cand_mask &= torch.tensor([int(t) not in special for t in encoded["input_ids"][1]], device=device)
            ref_vec = F.normalize(hidden[0][ref_mask], dim=-1)
            cand_vec = F.normalize(hidden[1][cand_mask], dim=-1)
            if ref_vec.numel() == 0 or cand_vec.numel() == 0:
                score = 0.0
            else:
                similarities = ref_vec @ cand_vec.T
                score = float(similarities.max(dim=1).values.clamp(min=0.0).mean().cpu())
        record["xlmr_bertscore_recall"] = score
        cache[item_id] = record
        append_cache(cache_path, record)
        if index % 50 == 0 or index == len(pending):
            print(f"Stage 3/4 XLM-R token recall: {index}/{len(pending)}", flush=True)
    del model, tokenizer
    gc.collect()


def safe_mean(values: Sequence[float], fallback: float = 0.0) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else fallback


def extract_nli_features(record: dict[str, Any]) -> dict[str, float]:
    scores = record.get("nli_feature_scores", {})
    whole = scores.get("whole", {})
    claims = [scores[k] for k in sorted(scores) if k.startswith("claim_")]
    negated = [scores[k] for k in sorted(scores) if k.startswith("negative_")]
    qa = scores.get("qa_answer", {})
    entail = [v.get("entailment", 0.0) for v in claims]
    contra = [v.get("contradiction", 0.0) for v in claims]
    neutral = [v.get("neutral", 0.0) for v in claims]
    margins = [e - c for e, c in zip(entail, contra)]
    features = {
        "whole_entailment_p": whole.get("entailment", 0.0),
        "whole_contradiction_p": whole.get("contradiction", 0.0),
        "whole_neutral_p": whole.get("neutral", 0.0),
        "claim_entailment_mean": safe_mean(entail),
        "claim_entailment_min": min(entail) if entail else 0.0,
        "claim_contradiction_max": max(contra) if contra else 0.0,
        "claim_contradiction_mean": safe_mean(contra),
        "claim_neutral_mean": safe_mean(neutral),
        "claim_margin_mean": safe_mean(margins),
        "claim_margin_min": min(margins) if margins else 0.0,
        "claim_support_rate": float(np.mean([e > c for e, c in zip(entail, contra)])) if entail else 0.0,
        "claim_contradiction_rate": float(np.mean([c > e for e, c in zip(entail, contra)])) if entail else 0.0,
        "negated_claim_contradiction_mean": safe_mean([v.get("contradiction", 0.0) for v in negated]),
        "qa_answer_entailment_p": qa.get("entailment", 0.0),
        "xlmr_bertscore_recall": float(record.get("xlmr_bertscore_recall", 0.0)),
        "e5_retrieval_margin": float(record.get("e5_retrieval_margin", 0.0)),
    }
    return {key: float(features[key]) for key in NLI_FEATURES}


def lexical_features(record: dict[str, Any]) -> dict[str, float]:
    answer = str(record.get("norm_answer", ""))
    question = str(record.get("norm_question", ""))
    evidence = str(record.get("evidence", ""))
    a_tokens = set(word_tokens(answer))
    e_tokens = set(word_tokens(evidence))
    q_tokens = set(word_tokens(question))
    intersection = len(a_tokens & e_tokens)
    precision = intersection / len(a_tokens) if a_tokens else 0.0
    recall = intersection / len(e_tokens) if e_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = intersection / len(a_tokens | e_tokens) if a_tokens | e_tokens else 0.0
    char_a = {answer[i:i + 3] for i in range(max(0, len(answer) - 2))}
    char_e = {evidence[i:i + 3] for i in range(max(0, len(evidence) - 2))}
    char_jaccard = len(char_a & char_e) / len(char_a | char_e) if char_a | char_e else 0.0
    numbers_a = set(re.findall(r"\d+(?:[.,]\d+)*", answer))
    numbers_e = set(re.findall(r"\d+(?:[.,]\d+)*", evidence))
    numeric = len(numbers_a & numbers_e) / len(numbers_a) if numbers_a else (1.0 if not numbers_e else 0.0)
    neg_a = bool(set(a_tokens) & NEGATION_WORDS)
    neg_e = bool(set(e_tokens) & NEGATION_WORDS)
    q_overlap = len(a_tokens & q_tokens) / len(a_tokens) if a_tokens else 0.0
    return {
        "token_fertility": float(record.get("e5_token_fertility", 0.0)),
        "word_jaccard_answer_evidence": float(jaccard),
        "char3_jaccard_answer_evidence": float(char_jaccard),
        "word_precision_answer_evidence": float(precision),
        "word_recall_answer_evidence": float(recall),
        "word_f1_answer_evidence": float(f1),
        "numeric_match_ratio": float(numeric),
        "negation_mismatch": float(neg_a != neg_e),
        "bge_reranker_margin": float(record.get("bge_reranker_margin", 0.0)),
        "question_answer_overlap": float(q_overlap),
    }


def full_feature_row(record: dict[str, Any]) -> dict[str, Any]:
    values = {}
    values.update(extract_nli_features(record))
    values.update({name: float(record.get("llm_logit_features", {}).get(name) or 0.0) for name in LOGIT_FEATURES})
    values.update(lexical_features(record))
    return values


def optimize_regime_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, float]:
    total_a = max(1, int(np.sum(tracks == "Track_A")))
    total_b = max(1, int(np.sum(tracks == "Track_B")))
    thresholds = {}
    for regime_value, name in ((True, "open"), (False, "closed")):
        mask = regimes == regime_value
        if not mask.any():
            thresholds[name] = 0.5
            continue
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities[mask])))
        candidates = np.unique(np.concatenate((candidates, (candidates[:-1] + candidates[1:]) / 2.0)))
        best = (float("inf"), float("inf"), 0.5)
        for threshold in candidates:
            pred = probabilities[mask] >= threshold
            local_tracks, local_y = tracks[mask], labels[mask]
            fp = int(np.sum((local_tracks == "Track_A") & pred))
            fn = int(np.sum((local_tracks == "Track_B") & ~pred))
            objective = 0.5 * fp / total_a + 0.5 * fn / total_b
            tie = abs(float(threshold) - 0.5)
            if (objective, tie, float(threshold)) < best:
                best = (objective, tie, float(threshold))
        thresholds[name] = best[2]
    return thresholds


def bhs_for_predictions(tracks: Sequence[str], predictions: Sequence[int]) -> dict[str, float]:
    track_array = np.asarray(tracks, dtype=object)
    pred_array = np.asarray(predictions, dtype=int)
    a = pred_array[track_array == "Track_A"]
    b = pred_array[track_array == "Track_B"]
    if not len(a) or not len(b):
        raise ValueError("BHS requires at least one item from each track")
    fpr = float(np.mean(a == 1))
    fnr = float(np.mean(b == 0))
    return {"FPR_A": fpr, "FNR_B": fnr, "BHS": (WEIGHTS[0] * fpr + WEIGHTS[1] * fnr) * 100.0}


def fit_meta_ensemble(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    x_train_dense = pd.DataFrame([full_feature_row(r) for r in train_records], columns=FEATURES).replace([np.inf, -np.inf], np.nan)
    x_eval_dense = pd.DataFrame([full_feature_row(r) for r in eval_records], columns=FEATURES).replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_dense = scaler.fit_transform(imputer.fit_transform(x_train_dense))
    eval_dense = scaler.transform(imputer.transform(x_eval_dense))
    train_text = [" ".join([r["norm_question"], r["norm_context"], r["norm_answer"]]) for r in train_records]
    eval_text = [" ".join([r["norm_question"], r["norm_context"], r["norm_answer"]]) for r in eval_records]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2, max_features=25000,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )
    train_char = vectorizer.fit_transform(train_text)
    eval_char = vectorizer.transform(eval_text)
    x_train = hstack([csr_matrix(train_dense), train_char], format="csr")
    x_eval = hstack([csr_matrix(eval_dense), eval_char], format="csr")
    y = np.asarray([int(r["ground_truth_label"]) for r in train_records], dtype=int)
    groups = np.asarray([str(r["group_key"]) for r in train_records], dtype=object)
    tracks = np.asarray([str(r["track"]) for r in train_records], dtype=object)
    regimes = np.asarray([bool(str(r["norm_context"]).strip()) for r in train_records], dtype=bool)
    if len(set(groups)) < N_FOLDS or len(set(y)) < 2:
        raise ValueError("Training data needs at least five question groups and both benchmark labels")
    c_oof: dict[float, np.ndarray] = {}
    for c_value in C_VALUES:
        per_seed = []
        for seed in SEEDS:
            oof = np.full(len(y), np.nan, dtype=float)
            splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fit_indices, valid_indices in splitter.split(x_train, y, groups):
                model = LogisticRegression(C=c_value, max_iter=2000, solver="liblinear", random_state=seed)
                model.fit(x_train[fit_indices], y[fit_indices])
                oof[valid_indices] = model.predict_proba(x_train[valid_indices])[:, 1]
            if np.isnan(oof).any():
                raise RuntimeError("Grouped OOF did not assign a prediction to every training row")
            per_seed.append(oof)
        c_oof[c_value] = np.mean(per_seed, axis=0)
        print(f"Stage 5/5 grouped OOF completed for C={c_value}", flush=True)
    selection = {}
    for c_value, probs in c_oof.items():
        thresholds = optimize_regime_thresholds(probs, y, tracks, regimes)
        predictions = np.asarray([
            int(p >= thresholds["open" if regime else "closed"])
            for p, regime in zip(probs, regimes)
        ])
        metrics = bhs_for_predictions(tracks, predictions)
        selection[c_value] = {"thresholds": thresholds, "metrics": metrics}
    chosen_c = min(C_VALUES, key=lambda c: (selection[c]["metrics"]["BHS"], abs(c - 1.0)))
    thresholds = selection[chosen_c]["thresholds"]
    eval_probability_runs = []
    final_models = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fit_indices, _ in splitter.split(x_train, y, groups):
            model = LogisticRegression(C=chosen_c, max_iter=2000, solver="liblinear", random_state=seed)
            model.fit(x_train[fit_indices], y[fit_indices])
            eval_probability_runs.append(model.predict_proba(x_eval)[:, 1])
            final_models.append(model)
    eval_probability = np.mean(eval_probability_runs, axis=0)
    eval_regimes = np.asarray([bool(str(r["norm_context"]).strip()) for r in eval_records], dtype=bool)
    predictions = np.asarray([
        int(p >= thresholds["open" if regime else "closed"])
        for p, regime in zip(eval_probability, eval_regimes)
    ], dtype=int)
    train_keys: dict[str, int] = {}
    for record in train_records:
        key = exact_memory_key(record["norm_question"], record["norm_answer"])
        label = int(record["ground_truth_label"])
        if key in train_keys and train_keys[key] != label:
            train_keys.pop(key, None)
        else:
            train_keys[key] = label
    memory_hits = 0
    for i, record in enumerate(eval_records):
        key = exact_memory_key(record["norm_question"], record["norm_answer"])
        if key in train_keys:
            predictions[i] = train_keys[key]
            memory_hits += 1
    metadata = {
        "selected_C": chosen_c,
        "threshold_open": thresholds["open"],
        "threshold_closed": thresholds["closed"],
        "training_oof_bhs_by_C": {str(k): v["metrics"]["BHS"] for k, v in selection.items()},
        "training_oof_thresholds_by_C": {str(k): v["thresholds"] for k, v in selection.items()},
        "seeds": list(SEEDS), "folds": N_FOLDS,
        "memory_overrides": memory_hits,
        "training_rows": len(train_records), "evaluation_rows": len(eval_records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "meta_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    predictions_df = pd.DataFrame({
        "id": [r["id"] for r in eval_records],
        "p_hallucinated_meta": eval_probability,
        "threshold_used": [thresholds["open" if regime else "closed"] for regime in eval_regimes],
        "predicted_label": predictions,
        "predicted_verdict": ["hallucinated" if p == 1 else "faithful" for p in predictions],
    })
    predictions_df.to_csv(output_dir / "meta_predictions.csv", index=False, encoding="utf-8-sig")
    return predictions, metadata, selection[chosen_c]


def exact_memory_key(question: str, answer: str) -> str:
    return sha256_text(canonicalize(question, DEFAULT_BANGLISH_MAP)[0] + "\n" + canonicalize(answer, DEFAULT_BANGLISH_MAP)[0])


def baseline_records(baseline_dir: Path, evaluation: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    outputs_dir = baseline_dir / "outputs"
    expected = set(evaluation.id.astype(str))
    all_models: dict[str, dict[str, dict[str, Any]]] = {}
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Baseline JSONL directory not found: {outputs_dir}")
    for path in outputs_dir.glob("*_zero_shot_track_a_b_first10_each.jsonl"):
        latest: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("id")) in expected:
                latest[str(row["id"])] = row
        if latest:
            model = next(iter(latest.values())).get("model_name") or path.name.split("_zero_shot_")[0]
            all_models[str(model)] = latest
    if not all_models:
        raise ValueError("No baseline JSONL files found for split track_a_b_first10_each")
    return all_models


def paired_comparison(
    evaluation: pd.DataFrame,
    pipeline_predictions: np.ndarray,
    baseline_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baselines = baseline_records(baseline_dir, evaluation)
    eval_ids = evaluation.id.astype(str).tolist()
    eval_track_by_id = dict(zip(eval_ids, evaluation.track.astype(str)))
    pipeline_by_id = dict(zip(eval_ids, pipeline_predictions.astype(int)))
    rows = []
    for model_name, records in sorted(baselines.items()):
        parsed_ids = {
            item_id for item_id, record in records.items()
            if record.get("parsed_verdict") in (0, 1) and not record.get("inference_error")
        }
        parsed_a = [i for i in parsed_ids if eval_track_by_id.get(i) == "Track_A"]
        parsed_b = [i for i in parsed_ids if eval_track_by_id.get(i) == "Track_B"]
        if not parsed_a or not parsed_b:
            raise ValueError(f"Baseline {model_name} lacks parsed predictions on one track")
        pre_a = np.asarray([int(records[i]["parsed_verdict"]) for i in parsed_a])
        pre_b = np.asarray([int(records[i]["parsed_verdict"]) for i in parsed_b])
        pre_fpr = float(np.mean(pre_a == 1))
        pre_fnr = float(np.mean(pre_b == 0))
        pre_bhs = 100.0 * (0.5 * pre_fpr + 0.5 * pre_fnr)
        matched_ids = parsed_a + parsed_b
        post_matched = bhs_for_predictions(
            [eval_track_by_id[i] for i in matched_ids], [pipeline_by_id[i] for i in matched_ids]
        )
        rows.append({
            "Baseline_Model": model_name,
            "Baseline_Parsed_A": len(parsed_a), "Baseline_Parsed_B": len(parsed_b),
            "Baseline_Parse_Failures": len(evaluation) - len(parsed_ids),
            "Baseline_FPR_A_pct": 100.0 * pre_fpr,
            "Baseline_FNR_B_pct": 100.0 * pre_fnr,
            "Baseline_BHS_pct": pre_bhs,
            "Pipeline_FPR_A_on_same_parsed_rows_pct": 100.0 * post_matched["FPR_A"],
            "Pipeline_FNR_B_on_same_parsed_rows_pct": 100.0 * post_matched["FNR_B"],
            "Pipeline_BHS_on_same_parsed_rows_pct": post_matched["BHS"],
            "Delta_BHS_pipeline_minus_baseline_same_rows": post_matched["BHS"] - pre_bhs,
        })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "pre_post_matched_comparison.csv", index=False, encoding="utf-8-sig")
    return frame, {"models": len(rows), "eval_ids": len(eval_ids)}


def append_manifest(
    output_dir: Path,
    train_path: Path,
    eval_path: Path,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    dropped: int,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    model_revisions: dict[str, str],
    ollama_digests: dict[str, str],
) -> None:
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_csv": str(train_path.resolve()), "train_sha256": file_sha256(train_path),
        "evaluation_csv": str(eval_path.resolve()), "evaluation_sha256": file_sha256(eval_path),
        "train_rows_available_after_question_group_exclusion": int(getattr(args, "filtered_train_rows_before_sampling", len(train))),
        "train_rows_used": len(train),
        "training_sample_size_argument": args.train_sample_size,
        "evaluation_rows": len(evaluation), "train_rows_removed_for_evaluation_question_overlap": dropped,
        "track_counts": evaluation.track.value_counts().to_dict(),
        "training_track_counts": train.track.value_counts().to_dict(),
        "training_sampling": {
            "requested_sample_size": args.train_sample_size,
            "selection": (
                "deterministic, approximately equal per-track sample; Track_A seed 42, Track_B seed 43"
                if args.train_sample_size is not None and args.train_sample_size < len(train)
                else "all leakage-filtered training rows"
            ),
        },
        "task_counts": evaluation.task.value_counts().to_dict(),
        "label_mapping": {"0": "faithful / Track_A", "1": "hallucinated / Track_B"},
        "weights": {"w_a": 0.5, "w_b": 0.5},
        "models": {
            "e5": args.e5_model, "reranker": args.reranker_model,
            "nli": args.nli_model, "xlmr": args.xlmr_model,
            "query_generator": args.qg_model, "qa": args.qa_model, "logit_judge": args.logit_model,
        },
        "model_cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "model_repository_revisions": model_revisions,
        "ollama_model_digests": ollama_digests,
        "retrieval": {"rrf_k": RRF_K, "top_k": RERANK_TOP_K, "evidence_top_n": EVIDENCE_TOP_N},
        "meta_classifier": metadata,
        "decoding": {"temperature": 0.0, "seed": 42, "qg_num_predict": 256, "qa_num_predict": 256, "logit_num_predict": 1},
        "feature_definitions": {
            "16_semantic": NLI_FEATURES,
            "3_logits": LOGIT_FEATURES,
            "10_lexical": LEXICAL_FEATURES,
            "char_tfidf": "character n-grams 2-5, max_features=25000, training-only vocabulary",
        },
        "limitations": [
            "The architecture images do not provide the original feature specification; this run uses the documented features above.",
            "The default Banglish map only transliterates listed words; unrecognized Latin words are retained and logged.",
            "External retrieval uses the supplied row context or leakage-filtered training contexts; no separate external corpus was supplied.",
            "mDeBERTa and XLM-R transfer scores are model features, not Bengali-specific calibrated probabilities.",
            "The pipeline detects hallucinations; it does not rewrite or correct candidate answers.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(no rows)_"
    columns = [str(c) for c in frame.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", " ") for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    header = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def write_pipeline_report(
    output_dir: Path,
    train: pd.DataFrame,
    available_train_rows: int,
    evaluation: pd.DataFrame,
    predictions: np.ndarray,
    metrics: dict[str, float],
    meta: dict[str, Any],
    results: pd.DataFrame,
    comparison: pd.DataFrame,
    stage_failures: dict[str, Any],
) -> None:
    predicted = np.asarray(predictions, dtype=int)
    labels = evaluation.ground_truth_label.astype(int).to_numpy()
    a_mask = evaluation.track.astype(str).to_numpy() == "Track_A"
    b_mask = evaluation.track.astype(str).to_numpy() == "Track_B"
    summary = pd.DataFrame([{
        "Rows": len(evaluation),
        "Track_A": int(a_mask.sum()),
        "Track_B": int(b_mask.sum()),
        "Track_A_false_alarms": int(np.sum(a_mask & (predicted == 1))),
        "Track_B_missed_hallucinations": int(np.sum(b_mask & (predicted == 0))),
        "FPR_A_pct": f"{metrics['FPR_A'] * 100:.2f}",
        "FNR_B_pct": f"{metrics['FNR_B'] * 100:.2f}",
        "BenHalluScore_pct": f"{metrics['BHS']:.2f}",
    }])
    comparison_view = comparison.copy()
    numeric_pct = [c for c in comparison_view.columns if c.endswith("_pct") or "same_rows" in c]
    for column in numeric_pct:
        if pd.api.types.is_numeric_dtype(comparison_view[column]):
            comparison_view[column] = comparison_view[column].map(lambda x: f"{x:.2f}")
    text = f"""# Updated BenHalluEval detector: 20-row Track-A/Track-B run

## Result

This run evaluates the same 20 input rows as the four-model baseline: 10 Track-A and 10 Track-B, all `bn_qa`. The updated detector predicted every row. The score uses the dataset label convention (0 = Track-A faithful, 1 = Track-B hallucinated):

{markdown_table(summary)}

Lower BenHalluScore is better. Track-A false alarms are predictions of hallucination on faithful examples. Track-B misses are predictions of faithful on hallucinated examples.

## Baseline comparison on identical parsed rows

The original judge JSONL files contain parse failures. To avoid assigning invented labels to those rows, each baseline BHS and its paired updated-detector BHS below use the exact same set of baseline-parsed IDs for that model. The full updated-detector result above still uses all 20 rows.

{markdown_table(comparison_view)}

The `Delta_BHS_pipeline_minus_baseline_same_rows` column is updated detector BHS minus baseline BHS. Negative values mean the updated detector made fewer weighted errors on the matched rows; positive values mean more.

## Reproducible method choices

- Training data: {len(train):,} rows used from {available_train_rows:,} eligible rows after removing every training row whose normalized question group appears in the evaluation set. An optional pilot sample is balanced by track with Track_A seed 42 and Track_B seed 43.
- Evaluation data: {len(evaluation)} rows; the normalized question groups are excluded from training.
- Retrieval: multilingual E5 dense retrieval plus in-file BM25, reciprocal-rank fusion with `k={RRF_K}`, then BGE reranking. A supplied row context is used directly; rows without context retrieve from the filtered training-context corpus. Exact evaluation-answer hashes are purged from that corpus.
- Claim and evidence QA models: `{QG_MODEL_DEFAULT}` and `{QA_MODEL_DEFAULT}` by default. The first-token binary-logit judge defaults to `{LOGIT_MODEL_DEFAULT}`. Decoding uses temperature 0 and seed 42.
- Feature model defaults: `{E5_MODEL_DEFAULT}`, `{RERANKER_MODEL_DEFAULT}`, `{NLI_MODEL_DEFAULT}`, and `{XLMR_MODEL_DEFAULT}`. The exact resolved Hugging Face commits and Ollama digests are recorded in `run_manifest.json`.
- Stage 5: grouped five-fold logistic regression, five random seeds, `C` in `{C_VALUES}`; model and open/closed-context thresholds are selected from training-only out-of-fold BHS.
- The pipeline is a hallucination detector. It classifies the existing answer; it does not rewrite or correct it. This result measures detector performance, not answer-quality improvement after correction.

## Audit and limitations

- Stage errors, claim-generation fallbacks, repetition-collapse counts/rates, and romanized-token audit are in `stage_failure_audit.json`.
- The diagrams do not define the 16 NLI features, 10 lexical features, the complete Banglish phoneme map, or the exact threshold-search objective. Their reproducible defaults and feature names are recorded in `run_manifest.json` and this script.
- Banglish normalization maps only reviewed words in the built-in/custom map; unknown Latin tokens are preserved and listed in the audit.
- The mDeBERTa/XLM-R signals are transfer features and are not Bengali-calibrated probabilities. The XLM-R feature is token-level max-cosine recall, an explicitly defined BERTScore-style proxy without baseline rescaling.
- This 20-row sample is exploratory and too small for a reliable general model ranking. Full thesis conclusions require the prescribed larger split and repeated evaluation.

## Files

- `update_pipeline_bhs.csv`: complete 20-row score and error counts.
- `pre_post_matched_comparison.csv`: per-baseline matched-row comparison.
- `meta_predictions.csv`: per-row labels, thresholds, and meta-classifier probabilities.
- `features_29_dense.csv`: extracted feature values for train and evaluation rows.
- `intermediate/five_stage_rows.jsonl`: resumable per-row stage outputs.
- `run_manifest.json`: data hashes, resolved model versions, model settings, and method details.
"""
    (output_dir / "update_pipeline_report.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval_csv", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--baseline_dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache_dir", type=Path, default=None, help="Hugging Face cache; defaults to HF_HOME or the standard cache")
    parser.add_argument("--banglish_map", type=Path, default=None, help="Reviewed JSON word map for romanized Bengali")
    parser.add_argument("--e5_model", default=E5_MODEL_DEFAULT)
    parser.add_argument("--reranker_model", default=RERANKER_MODEL_DEFAULT)
    parser.add_argument("--nli_model", default=NLI_MODEL_DEFAULT)
    parser.add_argument("--xlmr_model", default=XLMR_MODEL_DEFAULT)
    parser.add_argument("--qg_model", default=QG_MODEL_DEFAULT)
    parser.add_argument("--qa_model", default=QA_MODEL_DEFAULT)
    parser.add_argument("--logit_model", default=LOGIT_MODEL_DEFAULT)
    parser.add_argument("--ollama_url", default=OLLAMA_DEFAULT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--train_sample_size", type=int, default=None, help="Optional deterministic, approximately balanced per-track pilot sample; omit for all leakage-filtered rows")
    parser.add_argument("--force_recompute", action="store_true", help="Delete the intermediate checkpoint before starting")
    return parser


def _run(args: argparse.Namespace) -> int:
    import torch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate = args.output_dir / "intermediate"
    cache_path = intermediate / "five_stage_rows.jsonl"
    if args.force_recompute and cache_path.exists():
        cache_path.unlink()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    banglish_map = load_banglish_map(args.banglish_map)
    train, evaluation, dropped, _ = prepare_data(args.train_csv, args.eval_csv, banglish_map)
    args.filtered_train_rows_before_sampling = len(train)
    if args.train_sample_size is not None:
        if args.train_sample_size < 10 or args.train_sample_size > len(train):
            raise ValueError(f"--train_sample_size must be between 10 and {len(train)}")
        if args.train_sample_size < len(train):
            track_a_size = args.train_sample_size // 2
            track_b_size = args.train_sample_size - track_a_size
            available_a = train.loc[train.track == "Track_A"]
            available_b = train.loc[train.track == "Track_B"]
            if track_a_size > len(available_a) or track_b_size > len(available_b):
                raise ValueError("The requested pilot size exceeds the rows available in one benchmark track")
            sample_a = available_a.sample(n=track_a_size, random_state=42)
            sample_b = available_b.sample(n=track_b_size, random_state=43)
            train = pd.concat([sample_a, sample_b]).sort_index().reset_index(drop=True)
    print(f"Leakage guard removed {dropped} training rows; train rows available={args.filtered_train_rows_before_sampling}, rows selected={len(train)}, eval rows={len(evaluation)}", flush=True)
    if len(evaluation) != 20 or evaluation.track.value_counts().to_dict() != {"Track_A": 10, "Track_B": 10}:
        print("Warning: the evaluation input differs from the expected 10+10 example sample.", flush=True)
    train = train.copy()
    evaluation = evaluation.copy()
    train["split"] = "train"
    evaluation["split"] = "evaluation"
    rows = pd.concat([train, evaluation], ignore_index=True)
    cache = load_cache(cache_path)
    for _, row in rows.iterrows():
        item_id = str(row.id)
        fingerprint = row_fingerprint(row.to_dict(), str(row.split))
        if item_id not in cache or cache[item_id].get("input_fingerprint") != fingerprint:
            cache.pop(item_id, None)
            cache[item_id] = {
                "id": item_id, "input_fingerprint": fingerprint, "split": str(row.split),
                "track": str(row.track), "task": str(row.task), "hallucination_type": str(row.hallucination_type),
                "ground_truth_label": int(row.ground_truth_label), "group_key": str(row.group_key),
                "norm_question": str(row.norm_question), "norm_context": str(row.norm_context), "norm_answer": str(row.norm_answer),
                "raw_question": str(row.question), "raw_context": str(row.context), "raw_answer": str(row.candidate_answer),
                "roman_map_rate": float(row.roman_map_rate), "roman_oov_tokens": row.roman_oov_tokens,
            }
            append_cache(cache_path, cache[item_id])
    ollama = OllamaClient(args.ollama_url)
    stage1_qg_and_logits(rows, cache, cache_path, ollama, args.qg_model, args.logit_model)
    stage2_retrieve(rows, cache, cache_path, train, evaluation, args.e5_model, args.reranker_model, args.cache_dir, device)
    stage3_qa(rows, cache, cache_path, ollama, args.qa_model)
    stage3_nli(rows, cache, cache_path, args.nli_model, args.cache_dir, device)
    stage3_xlmr_recall(rows, cache, cache_path, args.xlmr_model, args.cache_dir, device)
    train_records = [cache[str(row.id)] for _, row in train.iterrows()]
    eval_records = [cache[str(row.id)] for _, row in evaluation.iterrows()]
    expected = [set(NLI_FEATURES), set(LOGIT_FEATURES), set(LEXICAL_FEATURES)]
    for record in train_records + eval_records:
        actual = full_feature_row(record)
        if set(actual) != set(FEATURES):
            raise RuntimeError("Feature extractor did not produce the configured 29 dense feature columns")
    feature_rows = [{**{"id": r["id"], "split": r["split"], "track": r["track"], "task": r["task"], "ground_truth_label": r["ground_truth_label"]}, **full_feature_row(r)} for r in train_records + eval_records]
    pd.DataFrame(feature_rows).to_csv(args.output_dir / "features_29_dense.csv", index=False, encoding="utf-8-sig")
    predictions, meta, selected = fit_meta_ensemble(train, evaluation, train_records, eval_records, args.output_dir)
    metrics = bhs_for_predictions(evaluation.track.astype(str).tolist(), predictions.tolist())
    prediction_df = pd.read_csv(args.output_dir / "meta_predictions.csv", encoding="utf-8-sig")
    prediction_df["track"] = evaluation.track.astype(str).tolist()
    prediction_df["ground_truth_label"] = evaluation.ground_truth_label.astype(int).tolist()
    prediction_df.to_csv(args.output_dir / "meta_predictions.csv", index=False, encoding="utf-8-sig")
    results = pd.DataFrame([{
        "System": "diagram_pipeline_5stage",
        "Rows": len(evaluation), "Track_A_Rows": int((evaluation.track == "Track_A").sum()),
        "Track_B_Rows": int((evaluation.track == "Track_B").sum()),
        "Track_A_FPR_pct": metrics["FPR_A"] * 100.0,
        "Track_B_FNR_pct": metrics["FNR_B"] * 100.0,
        "BenHalluScore_pct": metrics["BHS"],
        "Stage_Error_Rows": sum(bool(r.get("qg_error") or r.get("qa_error") or r.get("llm_logit_error")) for r in eval_records),
        "QG_Fallback_Rows": sum(bool(r.get("qg_fallback")) for r in eval_records),
        "QG_Collapse_Rows": sum(bool(r.get("qg_collapsed")) for r in eval_records),
        "QA_Collapse_Rows": sum(bool(r.get("qa_collapsed")) for r in eval_records),
        "Logit_Error_Rows": sum(bool(r.get("llm_logit_error")) for r in eval_records),
        "Selected_C": meta["selected_C"], "Threshold_open": meta["threshold_open"],
        "Threshold_closed": meta["threshold_closed"], "Memory_Overrides": meta["memory_overrides"],
    }])
    results.to_csv(args.output_dir / "update_pipeline_bhs.csv", index=False, encoding="utf-8-sig")
    comparison, _ = paired_comparison(evaluation, predictions, args.baseline_dir, args.output_dir)
    full_row = {"System": "diagram_pipeline_5stage_full20", **metrics, "Track_A_N": 10, "Track_B_N": 10}
    pd.DataFrame([full_row]).to_csv(args.output_dir / "update_pipeline_bhs_fractional.csv", index=False, encoding="utf-8-sig")
    collapse_by_model_and_stage: dict[str, dict[str, dict[str, int]]] = {}
    for model_name, stage_name, field in (
        (args.qg_model, "claim_generation", "qg_collapsed"),
        (args.qa_model, "evidence_qa", "qa_collapsed"),
    ):
        collapse_by_model_and_stage.setdefault(model_name, {})[stage_name] = {
            "collapsed_rows": sum(bool(r.get(field)) for r in eval_records),
            "rows": len(eval_records),
        }
    stage_failures = {
        "qg_fallback_rows": sum(bool(r.get("qg_fallback")) for r in eval_records),
        "qg_collapse_rows": sum(bool(r.get("qg_collapsed")) for r in eval_records),
        "qg_collapse_rate_pct": 100.0 * sum(bool(r.get("qg_collapsed")) for r in eval_records) / max(len(eval_records), 1),
        "qa_error_rows": sum(bool(r.get("qa_error")) for r in eval_records),
        "qa_collapse_rows": sum(bool(r.get("qa_collapsed")) for r in eval_records),
        "qa_collapse_rate_pct": 100.0 * sum(bool(r.get("qa_collapsed")) for r in eval_records) / max(len(eval_records), 1),
        "llm_logit_error_rows": sum(bool(r.get("llm_logit_error")) for r in eval_records),
        "collapse_by_model_and_stage": collapse_by_model_and_stage,
        "romanized_oov_tokens": sorted({token for r in eval_records for token in r.get("roman_oov_tokens", [])}),
    }
    (args.output_dir / "stage_failure_audit.json").write_text(json.dumps(stage_failures, ensure_ascii=False, indent=2), encoding="utf-8")
    model_revisions = {}
    for label, model_name in (("e5", args.e5_model), ("reranker", args.reranker_model), ("nli", args.nli_model), ("xlmr", args.xlmr_model)):
        model_revisions[label] = resolve_hf_revision(model_name, args.cache_dir)
    ollama_digests = ollama.model_digests([args.qg_model, args.qa_model, args.logit_model])
    append_manifest(args.output_dir, args.train_csv, args.eval_csv, train, evaluation, dropped, args, meta, model_revisions, ollama_digests)
    write_pipeline_report(args.output_dir, train, args.filtered_train_rows_before_sampling, evaluation, predictions, metrics, meta, results, comparison, stage_failures)
    print("\nUpdated pipeline BHS:")
    print(results.to_string(index=False))
    print("\nMatched-row baseline comparison:")
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {args.output_dir.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, FileNotFoundError, RuntimeError, urllib.error.URLError, pd.errors.ParserError) as exc:
        print(f"update_evaluation.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
