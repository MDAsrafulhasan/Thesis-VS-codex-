"""Offline, label-free neural features for the corrected thesis architecture.

Public API
----------
NeuralStages(cache_dir, device='cpu', threads=4).enrich(records, progress=None)
    Mutates/returns records in input order. ``progress(record)`` is called after
    each completed phase, including initialization and feature aggregation. The
    caller owns run-specific JSONL serialization (including torn-line repair),
    and should reuse these judge-independent records for all four judges.
feature_vector(record, judge_features)
    Returns exactly 30 entries: SEMANTIC_FEATURES + JUDGE_FEATURES +
    SURFACE_FEATURES. None is intentional: use a training-fitted imputer WITH
    missing indicators, never an implicit zero fill.

Evidence protocol
-----------------
Only the current record's supplied context is indexed; questions, answers,
other records, benchmark training contexts and saved predictions are NEVER an
alternative corpus. Retrieval uses E5 mean pooling with query:/passage: prefixes
and BM25Index(word_tokens) from the main-guarded update_evaluation.py. For EACH
claim and polarity, sparse/dense rankings are fused by RRF(k=60). A balanced
round-robin merge produces at most 20 unique candidates. BGE scores each
candidate against BOTH polarity queries; another balanced merge selects at most
four unique passages per claim. Polarity scores/ranks remain separate. NLI runs
independently on each selected passage, never on concatenated evidence, for the
original and supplied negative claim. Per-label evidence maxima are diagnostic
scores, NOT a normalized probability triple. Whole-answer NLI is disabled by
default; when enabled it also scores each selected passage independently before
maxima (its three exported values need not sum to one).

Semantic feature definitions (fixed order, 16)
---------------------------------------------
whole_{entailment,contradiction,neutral}_p: optional per-label evidence maxima.
claim_entailment_mean/min/smoothmin: across claims' max-evidence entailment.
claim_contradiction_mean/max/smoothmax: across claims' max-evidence contradiction.
claim_neutral_mean: mean of claims' max-evidence neutral scores.
claim_margin_mean: mean(max entailment - max contradiction), per claim.
negated_claim_contradiction_mean: mean max-evidence negative-claim contradiction.
qa_answer_entailment_p/qa_answer_contradiction_p: unavailable (None): QA is NOT
    implemented here, nor is token overlap relabeled as QA consistency.
xlmr_token_recall_proxy: optional mean candidate-token best evidence-token cosine,
    with special/padding tokens excluded; NOT calibrated BERTScore or a QA metric.
e5_polarity_margin_mean: mean over claims of best positive-query cosine minus
    best negative-query cosine among that claim's selected evidence.
Smooth/hard extrema use pipeline_v2_math.smooth_extrema(entails, contrads,
tau=stages.smooth_tau, weights=None), with default tau=0.15 and uniform weights.
Its smin_ent/smax_con/emin/cmax map to the four smooth/hard feature names.
Missing claims make paired extrema/routing unavailable rather than silently
averaging only easy/covered claims. record['nli_extrema'] retains that exact
four-key dict. record['route_inputs'] can be passed directly to math.route(**...):
has_context, is_math, entails, contrads, tau. Incomplete coverage gives empty
paired arrays, forcing full verification. record['nli_coverage'] explains why.

Surface feature definitions (10)
--------------------------------
E5 subwords per answer word; answer/evidence word Jaccard, raw-character trigram
Jaccard, unique-word precision/recall/F1; signed-number-set answer recall;
boolean negation-presence disagreement; mean within-polarity BGE top1-top2 logit
margin; unique-answer-word recall against the question. Evidence is the unique
union of selected passages across claims. These are lexical diagnostics, not
proof of correctness. Undefined denominators, no evidence, disabled models and
unavailable QA yield None and explicit missing_flags/reasons.

All pretrained constructors receive a resolved local snapshot directory,
local_files_only=True and trust_remote_code=False. No model loads at import or
construction. Phases load/unload one model at a time; float32, batch size two,
max length 512, caller-selected CPU threads. No training, API calls or judges.
Set ``stages.enable_xlmr = True`` and/or ``stages.enable_whole_nli = True`` before
enrich to enable these optional computations. Changes invalidate checkpoints.
The math module's dual_views dict is logged as text_views[field]['native'] and
['code']; both are NFKC/whitespace canonical views, not lossless raw text. With
no reviewed map, native == code (text_view_identity_fallback logs this per field).
Retrieval always preserves raw text: no guessed Banglish transduction or
question-as-evidence fallback is used.

Checkpoint compatibility includes raw inputs, model snapshot paths/configs,
options and this module's source hash. Matching completion markers can resume
ONLY intermediate features generated by this implementation, not predictions.
Do not share checkpoints between different scientific protocols. Keep model and
third-party licenses with experiment provenance; this module redistributes none.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

if __package__:
    from .update_evaluation import BM25Index, nli_label_indices, word_tokens
else:
    from update_evaluation import BM25Index, nli_label_indices, word_tokens

SEMANTIC_FEATURES = (
    "whole_entailment_p", "whole_contradiction_p", "whole_neutral_p",
    "claim_entailment_mean", "claim_entailment_min", "claim_entailment_smoothmin",
    "claim_contradiction_mean", "claim_contradiction_max", "claim_contradiction_smoothmax",
    "claim_neutral_mean", "claim_margin_mean", "negated_claim_contradiction_mean",
    "qa_answer_entailment_p", "qa_answer_contradiction_p",
    "xlmr_token_recall_proxy", "e5_polarity_margin_mean",
)
JUDGE_FEATURES = ("p_hall", "hard_vote", "margin", "jsd")
SURFACE_FEATURES = (
    "token_fertility", "word_jaccard_answer_evidence", "char3_jaccard_answer_evidence",
    "word_precision_answer_evidence", "word_recall_answer_evidence", "word_f1_answer_evidence",
    "numeric_match_ratio", "negation_mismatch", "bge_reranker_margin", "question_answer_overlap",
)
_MODELS = {
    "e5": "intfloat/multilingual-e5-large",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "nli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "xlmr": "FacebookAI/xlm-roberta-base",
}
_FORBIDDEN = {
    "label", "labels", "ground_truth_label", "gold", "gold_answer", "correct_answer",
    "track", "hallucination_type", "prediction", "predictions", "saved_predictions",
    "hallucinated_answer", "hallucinated_summary", "hallucinated_chain",
}
_NEGATIONS = {"না", "নয়", "নেই", "নাই", "নহে", "not", "never", "no", "without"}
_NUMBER = re.compile(r"[+\-−]?\d+(?:[.,]\d+)*(?:/\d+(?:[.,]\d+)*)?%?")
_LABELS = ("entailment", "contradiction", "neutral")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite feature value: {value!r}")
    return result


def _mean(values: list[float | None]) -> float | None:
    # Missing coverage must not turn into a high score on the surviving claims.
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _math_module():
    if __package__:
        from . import pipeline_v2_math
    else:
        import pipeline_v2_math
    return pipeline_v2_math


def _reject_leakage(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN:
                raise ValueError(f"Scoring/gold/prediction field is forbidden at inference: {key}")
            _reject_leakage(child)
    elif isinstance(value, list):
        for child in value:
            _reject_leakage(child)


def _balanced(positive: list[dict], negative: list[dict], limit: int) -> list[dict]:
    result, seen = [], set()
    for i in range(max(len(positive), len(negative))):
        for ranking in (positive, negative):
            if i < len(ranking) and ranking[i]["doc_id"] not in seen:
                doc = ranking[i]
                result.append(doc)
                seen.add(doc["doc_id"])
                if len(result) == limit:
                    return result
    return result


def _passages(context: str, row_id: str) -> list[dict]:
    # Bounded raw-text windows: no silent loss beyond a model's first 512 tokens.
    # Long unspaced text is also bounded; model-token truncation is logged later.
    result = []
    for sentence in re.split(r"(?<=[।!?])\s+|(?<=\.)\s+(?=[A-Z])|\n+", context):
        if not sentence.strip():
            continue
        for start in range(0, len(sentence), 600):
            text = sentence[start:start + 800].strip()
            if text:
                result.append({"doc_id": f"{row_id}:context:{len(result)}", "text": text,
                               "source": "record_context", "sha256": _digest(text)})
            if start + 800 >= len(sentence):
                break
    return result


def _selected(record: dict) -> list[dict]:
    found = {}
    for claim in record["claims"]:
        for evidence in claim.get("evidence", []):
            found.setdefault(evidence["doc_id"], evidence)
    return list(found.values())


def feature_vector(record: dict, judge_features: dict) -> list[float | None]:
    """16 semantic + 4 external judge + 10 surface; record missing indicators.

    Missing judge keys are unavailable, not a faithful verdict. This function
    never executes Stage 4. Persist missing_flags after calling it if desired.
    """
    semantic = record.get("semantic_features", {})
    surface = record.get("surface_features", {})
    values = ([_number(semantic.get(k)) for k in SEMANTIC_FEATURES]
              + [_number(judge_features.get(k)) for k in JUDGE_FEATURES]
              + [_number(surface.get(k)) for k in SURFACE_FEATURES])
    flags = record.setdefault("missing_flags", {})
    for name, value in zip(SEMANTIC_FEATURES + JUDGE_FEATURES + SURFACE_FEATURES, values):
        flags[name] = value is None
    return values


class NeuralStages:
    """Sequential, resumable, strictly local neural feature extraction."""

    def __init__(self, cache_dir, device: str = "cpu", threads: int = 4):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.device = device
        self.threads = threads
        self.enable_whole_nli = False
        self.enable_xlmr = False
        self.batch_size = 2
        self.max_length = 512
        self.smooth_tau = 0.15
        self.model_dirs: dict[str, Path] = {}

    def _resolve(self, stage: str) -> Path:
        """Resolve an explicit HF snapshot; never pass a repository ID to HF."""
        model_id = _MODELS[stage]
        root = self.cache_dir / ("models--" + model_id.replace("/", "--"))
        ref = root / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            if not revision or Path(revision).name != revision or revision in {".", ".."}:
                raise ValueError(f"Invalid local HF revision: {ref}")
            snapshot = root / "snapshots" / revision
        else:
            snapshots = sorted(p for p in (root / "snapshots").glob("*") if p.is_dir())
            if len(snapshots) != 1:
                raise FileNotFoundError(f"Need refs/main or exactly one local snapshot for {model_id}: {root}")
            snapshot = snapshots[0]
        if not (snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Missing local model config: {snapshot}")
        self.model_dirs[stage] = snapshot.resolve()
        return self.model_dirs[stage]

    def _load(self, stage: str):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(self.threads)
        path = str(self.model_dirs.get(stage) or self._resolve(stage))
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        constructor = AutoModel if stage in {"e5", "xlmr"} else AutoModelForSequenceClassification
        model = constructor.from_pretrained(path, torch_dtype=torch.float32, **kwargs)
        return tokenizer, model.to(self.device).eval()

    def _cleanup(self):
        gc.collect()
        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    def _encode(self, tokenizer, model, texts: list[str]):
        import torch
        import torch.nn.functional as F

        vectors = []
        for start in range(0, len(texts), self.batch_size):
            inputs = tokenizer(texts[start:start + self.batch_size], padding=True,
                               truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(F.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(vectors).numpy()

    def _pairs(self, tokenizer, model, pairs: list[tuple[str, str]], nli=False):
        import torch

        result = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            inputs = tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True,
                               truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = model(**inputs).logits.float()
                if nli:
                    result.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                else:
                    if logits.ndim != 2 or logits.shape[1] != 1:
                        raise ValueError("BGE reranker must expose exactly one relevance logit")
                    result.extend(logits[:, 0].cpu().tolist())
        return result

    def _retrieval(self, record, tokenizer, model):
        import numpy as np

        docs = _passages(record["context"], str(record["id"]))
        answer = record["candidate_answer"]
        words = word_tokens(answer)
        record["e5_token_fertility"] = (len(tokenizer.encode(answer, add_special_tokens=False)) / len(words)
                                         if words else None)
        if not docs or not record["claims"]:
            return
        dense = self._encode(tokenizer, model, ["passage: " + d["text"] for d in docs])
        bm25 = BM25Index([d["text"] for d in docs])
        for claim in record["claims"]:
            rankings = {}
            for polarity in ("positive", "negative"):
                query = claim[polarity + "_query"]
                if not query.strip():
                    rankings[polarity] = []
                    continue
                vector = self._encode(tokenizer, model, ["query: " + query])[0]
                dense_scores = dense @ vector
                sparse_scores = bm25.scores(query)
                dense_order = np.argsort(-dense_scores, kind="stable")[:20]
                sparse_order = [i for i in np.argsort(-sparse_scores, kind="stable") if sparse_scores[i] > 0][:20]
                dr = {int(i): rank for rank, i in enumerate(dense_order, 1)}
                sr = {int(i): rank for rank, i in enumerate(sparse_order, 1)}
                candidates = []
                for i in sorted(set(dr) | set(sr)):
                    rrf = (1 / (60 + dr[i]) if i in dr else 0) + (1 / (60 + sr[i]) if i in sr else 0)
                    candidates.append({**docs[i], "rrf_score": rrf, "e5_score": float(dense_scores[i]),
                                       "bm25_score": float(sparse_scores[i]),
                                       "dense_rank": dr.get(i), "sparse_rank": sr.get(i)})
                rankings[polarity] = sorted(candidates, key=lambda d: (-d["rrf_score"], d["doc_id"]))[:20]
            claim["retrieval_by_polarity"] = rankings
            merged = _balanced(rankings["positive"], rankings["negative"], 20)
            by_polarity = {p: {d["doc_id"]: d for d in ranks} for p, ranks in rankings.items()}
            claim["evidence_candidates"] = [
                {"doc_id": d["doc_id"], "text": d["text"], "source": d["source"], "sha256": d["sha256"],
                 "retrieval": {p: by_polarity[p].get(d["doc_id"]) for p in by_polarity}}
                for d in merged
            ]

    def _rerank(self, record, tokenizer, model):
        for claim in record["claims"]:
            candidates = claim["evidence_candidates"]
            rankings = {}
            for polarity in ("positive", "negative"):
                query = claim[polarity + "_query"]
                scores = self._pairs(tokenizer, model, [(query, d["text"]) for d in candidates]) if query.strip() else []
                rankings[polarity] = sorted(
                    [{**d, "reranker_score": float(s)} for d, s in zip(candidates, scores)],
                    key=lambda d: (-d["reranker_score"], d["doc_id"]))
            claim["reranked_by_polarity"] = rankings
            chosen = _balanced(rankings["positive"], rankings["negative"], 4)
            scores = {p: {d["doc_id"]: d["reranker_score"] for d in rank} for p, rank in rankings.items()}
            claim["evidence"] = [
                {**{k: v for k, v in d.items() if k != "reranker_score"},
                 "reranker_scores": {p: scores[p].get(d["doc_id"]) for p in scores}}
                for d in chosen
            ]

    def _nli(self, record, tokenizer, model, indices):
        for claim in record["claims"]:
            evidence = claim["evidence"]
            for key, hypothesis in (("nli", claim["claim"]), ("negative_nli", claim["negative_claim"])):
                values = self._pairs(tokenizer, model, [(d["text"], hypothesis) for d in evidence], nli=True) if hypothesis.strip() else []
                for d, scores in zip(evidence, values):
                    d[key] = {name: float(scores[index]) for name, index in zip(_LABELS, indices)}
                claim[key + "_maxima"] = {
                    name: max((d[key][name] for d in evidence if key in d), default=None) for name in _LABELS
                }
        record["whole_nli"] = None
        if self.enable_whole_nli and record["candidate_answer"].strip():
            evidence = _selected(record)
            values = self._pairs(tokenizer, model, [(d["text"], record["candidate_answer"]) for d in evidence], nli=True)
            record["whole_nli_per_evidence"] = [
                {"doc_id": d["doc_id"], **{name: float(s[index]) for name, index in zip(_LABELS, indices)}}
                for d, s in zip(evidence, values)
            ]
            record["whole_nli"] = {name: max((v[name] for v in record["whole_nli_per_evidence"]), default=None)
                                   for name in _LABELS}

    def _xlmr(self, record, tokenizer, model):
        import torch
        import torch.nn.functional as F

        candidate = record["candidate_answer"]
        evidence = _selected(record)
        if not candidate.strip() or not evidence:
            record["xlmr_token_recall_proxy"] = None
            return

        def tokens(text):
            encoded = tokenizer(text, truncation=True, max_length=self.max_length,
                                return_special_tokens_mask=True, return_tensors="pt").to(self.device)
            special = encoded.pop("special_tokens_mask").bool()[0]
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state.float()[0]
            keep = encoded["attention_mask"][0].bool() & ~special
            return F.normalize(hidden[keep], dim=-1)

        cand = tokens(candidate)
        best = None
        for doc in evidence:
            ref = tokens(doc["text"])
            if cand.numel() and ref.numel():
                current = (cand @ ref.T).max(dim=1).values
                best = current if best is None else torch.maximum(best, current)
        record["xlmr_token_recall_proxy"] = float(best.mean().cpu()) if best is not None else None

    def _features(self, record):
        maths = _math_module()
        claims = record["claims"]
        entail = [c.get("nli_maxima", {}).get("entailment") for c in claims]
        contra = [c.get("nli_maxima", {}).get("contradiction") for c in claims]
        neutral = [c.get("nli_maxima", {}).get("neutral") for c in claims]
        neg = [c.get("negative_nli_maxima", {}).get("contradiction") for c in claims]
        margins = [e - c if e is not None and c is not None else None for e, c in zip(entail, contra)]
        covered = [e is not None and c is not None for e, c in zip(entail, contra)]
        complete = bool(covered) and all(covered)
        route_entail, route_contra = (entail, contra) if complete else ([], [])
        extrema = maths.smooth_extrema(route_entail, route_contra, tau=self.smooth_tau, weights=None)
        record["nli_extrema"] = {key: _number(extrema[key]) for key in ("smin_ent", "smax_con", "emin", "cmax")}
        record["nli_coverage"] = {"complete": complete, "claims_total": len(claims), "claims_scored": sum(covered)}
        record["route_inputs"] = {
            "has_context": bool(record["context"].strip()), "is_math": record["task"] == "math",
            "entails": route_entail, "contrads": route_contra, "tau": self.smooth_tau,
        }
        whole = record.get("whole_nli") or {}
        polarity_margins = []
        bge_margins = []
        for claim in claims:
            scores = {}
            for p in ("positive", "negative"):
                vals = [d["retrieval"][p]["e5_score"] for d in claim["evidence"] if d["retrieval"].get(p)]
                scores[p] = max(vals) if vals else None
                ranked = claim.get("reranked_by_polarity", {}).get(p, [])
                bge_margins.append(ranked[0]["reranker_score"] - ranked[1]["reranker_score"] if len(ranked) >= 2 else None)
            polarity_margins.append(scores["positive"] - scores["negative"]
                                    if all(v is not None for v in scores.values()) else None)
        semantic = {
            "whole_entailment_p": whole.get("entailment"), "whole_contradiction_p": whole.get("contradiction"),
            "whole_neutral_p": whole.get("neutral"), "claim_entailment_mean": _mean(entail),
            "claim_entailment_min": record["nli_extrema"]["emin"],
            "claim_entailment_smoothmin": record["nli_extrema"]["smin_ent"],
            "claim_contradiction_mean": _mean(contra), "claim_contradiction_max": record["nli_extrema"]["cmax"],
            "claim_contradiction_smoothmax": record["nli_extrema"]["smax_con"],
            "claim_neutral_mean": _mean(neutral), "claim_margin_mean": _mean(margins),
            "negated_claim_contradiction_mean": _mean(neg),
            "qa_answer_entailment_p": None, "qa_answer_contradiction_p": None,
            "xlmr_token_recall_proxy": record.get("xlmr_token_recall_proxy"),
            "e5_polarity_margin_mean": _mean(polarity_margins),
        }
        answer, question = record["candidate_answer"], record["question"]
        evidence = "\n".join(d["text"] for d in _selected(record))
        a, e, q = (set(word_tokens(t)) for t in (answer, evidence, question))
        overlap = len(a & e)
        valid = bool(a and e)
        precision = overlap / len(a) if valid else None
        recall = overlap / len(e) if valid else None
        ca = {answer[i:i + 3] for i in range(max(0, len(answer) - 2))}
        ce = {evidence[i:i + 3] for i in range(max(0, len(evidence) - 2))}

        def numbers(text):
            # Unicode decimal digits fold to ASCII, signs/fractions remain distinct.
            text = "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in text)
            return {n.replace("−", "-") for n in _NUMBER.findall(text)}

        na, ne = numbers(answer), numbers(evidence)
        surface = {
            "token_fertility": record.get("e5_token_fertility"),
            "word_jaccard_answer_evidence": overlap / len(a | e) if valid else None,
            "char3_jaccard_answer_evidence": len(ca & ce) / len(ca | ce) if ca and ce else None,
            "word_precision_answer_evidence": precision, "word_recall_answer_evidence": recall,
            "word_f1_answer_evidence": (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if valid else None,
            "numeric_match_ratio": len(na & ne) / len(na) if na and evidence else None,
            "negation_mismatch": float(bool(a & _NEGATIONS) != bool(e & _NEGATIONS)) if valid else None,
            "bge_reranker_margin": _mean(bge_margins),
            "question_answer_overlap": len(a & q) / len(a) if a and q else None,
        }
        record["semantic_features"] = {k: _number(semantic[k]) for k in SEMANTIC_FEATURES}
        record["surface_features"] = {k: _number(surface[k]) for k in SURFACE_FEATURES}
        record["missing_flags"] = {k: v is None for k, v in {**semantic, **surface}.items()}
        record["missing_reasons"] = {k: "undefined_or_incomplete_evidence_coverage" for k, v in {**semantic, **surface}.items() if v is None}
        for name in ("qa_answer_entailment_p", "qa_answer_contradiction_p"):
            record["missing_reasons"][name] = "qa_unavailable_not_implemented_in_this_stage"
        if not self.enable_whole_nli:
            for name in SEMANTIC_FEATURES[:3]:
                record["missing_reasons"][name] = "whole_nli_disabled"
        if not self.enable_xlmr:
            record["missing_reasons"]["xlmr_token_recall_proxy"] = "xlmr_proxy_disabled"

    def enrich(self, records: list[dict], progress: Callable[[dict], None] | None = None) -> list[dict]:
        """Enrich in place; callback once per record after each completed phase.

        Callback exceptions propagate; persist a copy immediately if using an
        asynchronous writer. Set options before calling. Resumption skips only
        phases with this module's matching input/protocol fingerprint. Runtime
        failures propagate instead of manufacturing scores or marking success.
        """
        if not isinstance(records, list):
            raise TypeError("records must be a list of dictionaries")
        maths = _math_module()
        ids = set()
        for record in records:
            _reject_leakage(record)
            for field in ("id", "question", "context", "candidate_answer", "task", "claims"):
                if field not in record:
                    raise ValueError(f"Missing required input field: {field}")
            if str(record["id"]) in ids:
                raise ValueError(f"Duplicate record id: {record['id']}")
            ids.add(str(record["id"]))
            for field in ("question", "context", "candidate_answer", "task"):
                if not isinstance(record[field], str):
                    raise TypeError(f"{field} must be a string (use empty context for unavailable evidence)")
            if not isinstance(record["claims"], list):
                raise TypeError("claims must be a list")
            for claim in record["claims"]:
                if not isinstance(claim, dict) or any(not isinstance(claim.get(k), str) for k in
                    ("claim", "positive_query", "negative_query", "negative_claim")):
                    raise ValueError("Each claim needs string claim/positive_query/negative_query/negative_claim")
        if not records:
            return records
        has_work = any(r["context"].strip() and r["claims"] for r in records)
        model_stages = ["e5", "reranker", "nli"] + (["xlmr"] if self.enable_xlmr else [])
        if has_work:
            for stage in model_stages:
                self._resolve(stage)
        protocol = {
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "math_sha256": hashlib.sha256(Path(maths.__file__).read_bytes()).hexdigest(),
            "models": {k: str(v) for k, v in self.model_dirs.items()},
            "whole_nli": self.enable_whole_nli, "xlmr": self.enable_xlmr,
            "smooth_tau": self.smooth_tau, "claim_weights": "uniform (weights=None)",
            "device": str(self.device), "threads": self.threads, "batch_size": self.batch_size,
            "max_length": self.max_length, "retrieval": "own_context_only_rrf60_balanced20_to4",
            "truncation": "512_tokens_per_input_or_pair; raw_context_windows_800_chars_stride600",
        }
        for record in records:
            raw = {k: record[k] for k in ("id", "question", "context", "candidate_answer", "task")}
            raw["claims"] = [{k: c[k] for k in ("claim", "positive_query", "negative_query", "negative_claim")}
                             for c in record["claims"]]
            fingerprint = _digest({"input": raw, "protocol": protocol})
            state = record.get("neural_stages", {})
            if state.get("fingerprint") != fingerprint:
                # Clear only this module's derived outputs, never raw fields.
                for key in ("semantic_features", "surface_features", "missing_flags", "missing_reasons",
                            "whole_nli", "whole_nli_per_evidence", "e5_token_fertility", "xlmr_token_recall_proxy",
                            "nli_extrema", "nli_coverage", "route_inputs"):
                    record.pop(key, None)
                for claim in record["claims"]:
                    for key in ("retrieval_by_polarity", "evidence_candidates", "reranked_by_polarity",
                                "evidence", "nli_maxima", "negative_nli_maxima"):
                        claim.pop(key, None)
                state = {"fingerprint": fingerprint, "completed": [], "protocol": protocol}
                record["neural_stages"] = state
            record["no_evidence"] = not bool(record["context"].strip())
            record["evidence_status"] = "no_evidence" if record["no_evidence"] else ("no_claims" if not record["claims"] else "context_available")
            record["qa_status"] = "unavailable: QA decoupled from QG and not executed here"
            record["text_views"] = {}
            for field in ("question", "context", "candidate_answer"):
                views = maths.dual_views(record[field])
                record["text_views"][field] = {"native": views["native"], "code": views["code"]}
            record["text_view_identity_fallback"] = {
                field: views["native"] == views["code"] for field, views in record["text_views"].items()
            }
            record["text_view_policy"] = "native/code canonical views with no reviewed map; raw retrieval; no learned Banglish transducer"
            for claim in record["claims"]:
                claim.setdefault("retrieval_by_polarity", {"positive": [], "negative": []})
                claim.setdefault("evidence_candidates", [])
                claim.setdefault("reranked_by_polarity", {"positive": [], "negative": []})
                claim.setdefault("evidence", [])
                claim.setdefault("nli_maxima", dict.fromkeys(_LABELS))
                claim.setdefault("negative_nli_maxima", dict.fromkeys(_LABELS))
            if progress:
                progress(record)

        for phase in ("retrieval", "rerank", "nli", "xlmr", "features"):
            pending = [r for r in records if phase not in r["neural_stages"]["completed"]]
            work = [r for r in pending if not r["no_evidence"] and r["claims"]]
            if phase == "xlmr" and not self.enable_xlmr:
                work = []
            tokenizer = model = None
            try:
                if phase != "features" and work:
                    stage = {"retrieval": "e5", "rerank": "reranker", "nli": "nli", "xlmr": "xlmr"}[phase]
                    tokenizer, model = self._load(stage)
                indices = None
                if phase == "nli" and model is not None:
                    labels = [str(v).casefold() for v in model.config.id2label.values()]
                    if model.config.num_labels != 3 or any(not any(name in label for label in labels) for name in _LABELS):
                        raise ValueError(f"NLI config must explicitly name all three labels: {model.config.id2label}")
                    indices = nli_label_indices(model.config)
                work_ids = {id(r) for r in work}
                for record in pending:
                    if phase == "features":
                        self._features(record)
                    elif id(record) in work_ids:
                        if phase == "retrieval":
                            self._retrieval(record, tokenizer, model)
                        elif phase == "rerank":
                            self._rerank(record, tokenizer, model)
                        elif phase == "nli":
                            self._nli(record, tokenizer, model, indices)
                        else:
                            self._xlmr(record, tokenizer, model)
                    record["neural_stages"]["completed"].append(phase)
                    if progress:
                        progress(record)
            finally:
                del model, tokenizer
                self._cleanup()
        return records


__all__ = ["NeuralStages", "SEMANTIC_FEATURES", "SURFACE_FEATURES", "JUDGE_FEATURES", "feature_vector"]
