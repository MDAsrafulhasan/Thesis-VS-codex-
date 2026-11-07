#!/usr/bin/env python3
"""Reproducible inference, parsing, and reporting for BenHalluEval and Dialectal_Bias."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd


SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 1024
PROMPT_MODES = ("zero_shot", "cot")
TASKS = ("bn_qa", "codemixed_qa", "summarization", "math")
HALLUCINATION_TYPES = (
    "Factualness",
    "Comprehension",
    "Specificity",
    "Inference",
    "Fabricated_Content",
    "Non_Factual_Addition",
    "Direct_Contradiction",
    "Arithmetic_Slip",
    "Formula_Misapplication",
    "Invalid_Deduction",
    "Hallucinated_Intermediate_Fact",
    "Semantic_Drift",
)
BENHALLU_COLUMNS = {
    "id", "task", "track", "hallucination_type", "question", "context",
    "candidate_answer", "ground_truth_label",
}
DIALECT_COLUMNS = {
    "id", "dialect", "question_type", "question", "response", "human_or_primary_score",
}
class JudgeBackend(Protocol):
    def generate(self, prompt: str) -> tuple[str, int, int]:
        """Return response text, prompt token count, and completion token count."""


def _binary_value(value: str) -> int | None:
    normalized = value.strip().strip("`*_ \t\r\n.,!?;:。؛،()[]{}\"'").casefold()
    if normalized in ("yes", "true", "হ্যাঁ"):
        return 1
    if normalized in ("no", "false", "না"):
        return 0
    return None


def _has_repetition_collapse(text: str) -> bool:
    words = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    n_words = len(words)
    if n_words < 20:
        return False
    # Check every possible consecutive block length of at least five words.
    # Four equal adjacent copies satisfy the requested collapse definition.
    for ngram_len in range(5, n_words // 4 + 1):
        last_start = n_words - 4 * ngram_len
        for start in range(last_start + 1):
            phrase = words[start : start + ngram_len]
            if all(words[start + k * ngram_len : start + (k + 1) * ngram_len] == phrase for k in range(1, 4)):
                return True
    return False


def parse_judge_output(
    text: str,
    completion_tokens: int | None = None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """Parse English/Bengali binary verdicts and flag repetition/token-limit collapse."""
    text = text or ""
    is_collapsed = _has_repetition_collapse(text)
    has_closing_verdict_tag = bool(re.search(r"</\s*verdict\s*>", text, flags=re.IGNORECASE))
    if completion_tokens is not None and completion_tokens >= max_tokens and not has_closing_verdict_tag:
        is_collapsed = True

    candidates: list[tuple[int, int]] = []
    # Structured verdict tags have highest priority. The final occurrence wins,
    # since a chain-of-thought answer can contain an earlier tentative verdict.
    for match in re.finditer(r"<verdict>\s*(.*?)\s*</verdict\s*>", text, flags=re.IGNORECASE | re.DOTALL):
        value = _binary_value(match.group(1))
        if value is not None:
            candidates.append((3, value))

    # JSON objects are parsed rather than searched as loose text, avoiding
    # accidental matches in explanatory prose.
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("hallucinated", "verdict", "is_hallucinated"):
            if key not in obj:
                continue
            value = obj[key]
            parsed = None
            if isinstance(value, bool):
                parsed = int(value)
            elif isinstance(value, (str, int)):
                parsed = _binary_value(str(value))
            if parsed is not None:
                candidates.append((2, parsed))
                break

    verdict_word = r"(Yes|No|True|False|હ্যাঁ|না)"
    for match in re.finditer(
        rf"(?:Final\s+Verdict|Verdict)\s*(?::|=|is)?\s*{verdict_word}",
        text,
        flags=re.IGNORECASE,
    ):
        value = _binary_value(match.group(1))
        if value is not None:
            candidates.append((1, value))

    if candidates:
        priority = max(item[0] for item in candidates)
        verdict = [item[1] for item in candidates if item[0] == priority][-1]
    else:
        verdict = None
    return {
        "parsed_verdict": verdict,
        "is_collapsed": bool(is_collapsed),
        "parse_failure": verdict is None,
    }


def parse_score_output(text: str) -> dict[str, Any]:
    """Parse a numeric 0--10 dialect score from a structured model response."""
    text = text or ""
    candidates: list[float] = []
    patterns = (
        r"<score>\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*</score\s*>",
        r"(?:Final\s+Score|Score)\s*(?::|=|is)\s*(-?(?:\d+(?:\.\d*)?|\.\d+))",
    )
    for pattern in patterns:
        candidates.extend(float(m.group(1)) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    score = candidates[-1] if candidates else None
    if score is not None and not (math.isfinite(score) and 0.0 <= score <= 10.0):
        score = None
    return {"parsed_score": score, "parse_failure": score is None}


def compute_bhs(
    preds_track_a: Iterable[int],
    preds_track_b: Iterable[int],
    w_a: float = 0.5,
    w_b: float = 0.5,
) -> dict[str, float]:
    """Compute Track-A false positive and Track-B false negative rates and BHS."""
    pred_a = np.asarray(list(preds_track_a), dtype=int)
    pred_b = np.asarray(list(preds_track_b), dtype=int)
    if pred_a.size == 0 or pred_b.size == 0:
        raise ValueError("BHS requires at least one parsed prediction from each track")
    if not np.isin(pred_a, (0, 1)).all() or not np.isin(pred_b, (0, 1)).all():
        raise ValueError("BHS predictions must be binary values 0 or 1")
    if not (math.isfinite(w_a) and math.isfinite(w_b) and w_a >= 0 and w_b >= 0):
        raise ValueError("BHS weights must be finite and nonnegative")
    if not math.isclose(w_a + w_b, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("BHS weights must sum to 1")
    fpr_a = float(np.mean(pred_a == 1))
    fnr_b = float(np.mean(pred_b == 0))
    return {"FPR_A": fpr_a, "FNR_B": fnr_b, "BHS": (w_a * fpr_a + w_b * fnr_b) * 100.0}


def compute_lins_ccc(x: Iterable[float], y: Iterable[float]) -> float:
    """Lin's concordance correlation coefficient using population moments."""
    arr_x = np.asarray(list(x), dtype=float)
    arr_y = np.asarray(list(y), dtype=float)
    if arr_x.ndim != 1 or arr_y.ndim != 1 or arr_x.size != arr_y.size or arr_x.size == 0:
        raise ValueError("CCC requires non-empty, one-dimensional arrays of equal length")
    if not np.isfinite(arr_x).all() or not np.isfinite(arr_y).all():
        raise ValueError("CCC arrays must contain only finite values")
    mean_x, mean_y = np.mean(arr_x), np.mean(arr_y)
    var_x, var_y = np.var(arr_x), np.var(arr_y)
    cov_xy = np.mean((arr_x - mean_x) * (arr_y - mean_y))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0.0:
        return 1.0 if np.array_equal(arr_x, arr_y) else float("nan")
    return float((2.0 * cov_xy) / denominator)


def compute_critical_bias_sensitivity(
    scores: Iterable[float], dialects: Iterable[str]
) -> dict[str, float | str]:
    """Return the anchor-to-Chittagong mean score drop, in score points."""
    values = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(dialects), dtype=str)
    if values.size == 0 or values.size != labels.size or not np.isfinite(values).all():
        raise ValueError("CBS requires equally sized, non-empty finite scores and dialect labels")
    available = set(labels.tolist())
    anchor = "Tangail" if "Tangail" in available else "Standard_Bengali"
    if anchor not in available or "Chittagong" not in available:
        raise ValueError("CBS requires Tangail or Standard_Bengali and Chittagong observations")
    anchor_mean = float(np.mean(values[labels == anchor]))
    chittagong_mean = float(np.mean(values[labels == "Chittagong"]))
    return {
        "anchor_variant": anchor,
        "anchor_mean": anchor_mean,
        "chittagong_mean": chittagong_mean,
        "CBS": anchor_mean - chittagong_mean,
    }


class MockJudge:
    """Deterministic baseline backend: always returns a No verdict / score 7."""

    def __init__(self, dialect_mode: bool = False):
        self.dialect_mode = dialect_mode

    def generate(self, prompt: str) -> tuple[str, int, int]:
        response = "<score>7</score>" if self.dialect_mode else "<verdict>No</verdict>"
        return response, len(prompt.split()), len(response.split())


class OllamaJudge:
    def __init__(self, model: str, endpoint: str):
        self.model = model
        self.endpoint = endpoint

    def generate(self, prompt: str) -> tuple[str, int, int]:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "seed": SEED, "num_predict": MAX_TOKENS},
        }).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (
            str(payload.get("response", "")),
            int(payload.get("prompt_eval_count", 0)),
            int(payload.get("eval_count", 0)),
        )


class TransformersJudge:
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("The transformers backend requires both torch and transformers") from exc
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if importlib.util.find_spec("accelerate") is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        self.model.eval()

    def generate(self, prompt: str) -> tuple[str, int, int]:
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            encoded = {"input_ids": inputs}
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            encoded["attention_mask"] = attention_mask.to(self.model.device)
        input_count = int(input_ids.shape[-1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                # Greedy decoding is deterministic and temperature-independent;
                # this is the temperature=0.0 equivalent for Transformers.
                do_sample=False,
                max_new_tokens=MAX_TOKENS,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, input_count:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, input_count, int(new_tokens.shape[-1])


def _make_backend(kind: str, model: str, endpoint: str, dialect_mode: bool) -> JudgeBackend:
    if kind == "mock":
        return MockJudge(dialect_mode=dialect_mode)
    if kind == "ollama":
        return OllamaJudge(model, endpoint)
    if kind == "transformers":
        return TransformersJudge(model)
    raise ValueError(f"Unsupported backend: {kind}")


def _build_primary_prompt(row: dict[str, Any], prompt_mode: str) -> str:
    context = row.get("context", "")
    question = row.get("question", "")
    answer = row.get("candidate_answer", "")
    if prompt_mode == "cot":
        instruction = (
            "Carefully compare the candidate answer with the supplied context and question. "
            "Explain your reasoning briefly, then give the final binary label in the required tag. "
        )
    else:
        instruction = "Judge the candidate answer using the supplied context and question. "
    return (
        "You are evaluating whether a Bengali or code-mixed answer contains hallucinated content. "
        "Use Yes when the answer is unsupported, fabricated, contradictory, or otherwise hallucinatory; "
        "use No when it is supported.\n\n"
        f"{instruction}\nQuestion: {question}\nContext: {context}\nCandidate answer: {answer}\n\n"
        "End with exactly one final tag: <verdict>Yes</verdict> or <verdict>No</verdict>."
    )


def _build_dialect_prompt(row: dict[str, Any], prompt_mode: str) -> str:
    instruction = (
        "Assess how well the response answers the question in its dialect, preserving meaning and factual usefulness. "
        "Reason step by step before the final score. "
        if prompt_mode == "cot"
        else "Assess how well the response answers the question in its dialect, preserving meaning and factual usefulness. "
    )
    return (
        "Rate the response on a 0 to 10 scale, using the same quality interpretation across dialects. "
        f"Dialect: {row['dialect']}\nQuestion type: {row['question_type']}\n"
        f"Question: {row['question']}\nResponse: {row['response']}\n"
        f"{instruction}End with exactly one score tag, for example <score>7.5</score>."
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "model"


def _recover_checkpoint(path: Path) -> set[str]:
    """Keep complete JSONL lines, truncate a torn tail, and return successful ids."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    offset = 0
    safe_offset = 0
    completed: set[str] = set()
    for line in data.splitlines(keepends=True):
        next_offset = offset + len(line)
        if not line.endswith(b"\n"):
            break
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
        if not isinstance(record, dict) or "id" not in record:
            break
        safe_offset = next_offset
        if not record.get("inference_error"):
            completed.add(str(record["id"]))
        offset = next_offset
    if safe_offset != len(data):
        with path.open("r+b") as handle:
            handle.truncate(safe_offset)
            handle.flush()
            os.fsync(handle.fileno())
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _infer_records(
    frame: pd.DataFrame,
    model_name: str,
    prompt_mode: str,
    backend: JudgeBackend,
    checkpoint: Path,
    dialect_mode: bool,
) -> None:
    completed = _recover_checkpoint(checkpoint)
    for row in frame.to_dict("records"):
        item_id = str(row["id"])
        if item_id in completed:
            continue
        prompt = _build_dialect_prompt(row, prompt_mode) if dialect_mode else _build_primary_prompt(row, prompt_mode)
        started = time.perf_counter()
        inference_error = None
        try:
            raw_response, prompt_tokens, completion_tokens = backend.generate(prompt)
        except Exception as exc:  # Persist the failure and let a later restart retry this id.
            raw_response = ""
            prompt_tokens = completion_tokens = 0
            inference_error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0
        if dialect_mode:
            parsed = parse_score_output(raw_response)
            is_collapsed = _has_repetition_collapse(raw_response) or (
                completion_tokens >= MAX_TOKENS and not re.search(r"</\s*score\s*>", raw_response, re.IGNORECASE)
            )
            record: dict[str, Any] = {
                "id": item_id,
                "task": row["question_type"],
                "track": None,
                "hallucination_type": None,
                "raw_prompt": prompt,
                "raw_response": raw_response,
                "parsed_verdict": None,
                "parsed_score": parsed["parsed_score"],
                "parse_failure": parsed["parse_failure"] or inference_error is not None,
                "is_collapsed": bool(is_collapsed),
                "latency_ms": latency_ms,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "model_name": model_name,
                "prompt_mode": prompt_mode,
                "dialect": row["dialect"],
                "human_or_primary_score": float(row["human_or_primary_score"]),
            }
        else:
            parsed = parse_judge_output(raw_response, completion_tokens=completion_tokens)
            record = {
                "id": item_id,
                "task": row["task"],
                "track": row["track"],
                "hallucination_type": row["hallucination_type"],
                "raw_prompt": prompt,
                "raw_response": raw_response,
                "parsed_verdict": parsed["parsed_verdict"],
                "parse_failure": parsed["parse_failure"] or inference_error is not None,
                "is_collapsed": parsed["is_collapsed"],
                "latency_ms": latency_ms,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "model_name": model_name,
                "prompt_mode": prompt_mode,
                "ground_truth_label": int(row["ground_truth_label"]),
            }
        if inference_error is not None:
            record["inference_error"] = inference_error
        _append_jsonl(checkpoint, record)
        if inference_error is not None:
            raise RuntimeError(
                f"Inference failed for id={item_id}; the failure is checkpointed and this id will be retried on restart: {inference_error}"
            )


def _read_checkpoint_records(outputs_dir: Path, split_name: str, valid_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not outputs_dir.exists():
        return records
    for path in sorted(outputs_dir.glob(f"*_*_{split_name}.jsonl")):
        _recover_checkpoint(path)
        latest_by_id: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("id")) in valid_ids:
                latest_by_id[str(row["id"])] = row
        records.extend(latest_by_id.values())
    return records


def _metric(records: list[dict[str, Any]]) -> dict[str, float | int]:
    parsed = [r for r in records if r.get("parsed_verdict") in (0, 1) and not r.get("inference_error")]
    a = [int(r["parsed_verdict"]) for r in parsed if r.get("track") == "Track_A"]
    b = [int(r["parsed_verdict"]) for r in parsed if r.get("track") == "Track_B"]
    fpr = float(np.mean(np.asarray(a) == 1)) if a else float("nan")
    fnr = float(np.mean(np.asarray(b) == 0)) if b else float("nan")
    bhs = (0.5 * fpr + 0.5 * fnr) * 100.0 if math.isfinite(fpr) and math.isfinite(fnr) else float("nan")
    total = len(records)
    return {
        "Track_A_FPR (%)": 100.0 * fpr if math.isfinite(fpr) else float("nan"),
        "Track_B_FNR (%)": 100.0 * fnr if math.isfinite(fnr) else float("nan"),
        "BenHalluScore (%)": bhs,
        "Collapse_Rate (%)": 100.0 * float(np.mean([bool(r.get("is_collapsed")) for r in records])) if total else float("nan"),
        "Parse_Failure_Rate (%)": 100.0 * float(np.mean([bool(r.get("parse_failure")) for r in records])) if total else float("nan"),
        "Avg_Latency_ms": float(np.mean([float(r.get("latency_ms", 0.0)) for r in records])) if total else float("nan"),
        "N_Rows": total,
        "N_Parsed": len(parsed),
    }


def _write_table(frame: pd.DataFrame, results_dir: Path, stem: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / f"{stem}.csv", index=False, encoding="utf-8")
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in display.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    (results_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _primary_tables(records: list[dict[str, Any]], results_dir: Path) -> None:
    model_modes = sorted({(str(r["model_name"]), str(r["prompt_mode"])) for r in records})
    table1_rows = []
    for model_name, mode in model_modes:
        model_rows = [r for r in records if r["model_name"] == model_name and r["prompt_mode"] == mode]
        for task in ("OVERALL", *TASKS):
            rows = model_rows if task == "OVERALL" else [r for r in model_rows if r.get("task") == task]
            table1_rows.append({"Model": model_name, "Prompt_Mode": mode, "Task": task, **_metric(rows)})
    table1 = pd.DataFrame(table1_rows, columns=[
        "Model", "Prompt_Mode", "Task", "Track_A_FPR (%)", "Track_B_FNR (%)", "BenHalluScore (%)",
        "Collapse_Rate (%)", "Parse_Failure_Rate (%)", "Avg_Latency_ms", "N_Rows", "N_Parsed",
    ])
    _write_table(table1, results_dir, "table1_overall_and_per_task_bhs")

    table2_rows = []
    for model_name, mode in model_modes:
        model_rows = [r for r in records if r["model_name"] == model_name and r["prompt_mode"] == mode]
        for hallucination_type in HALLUCINATION_TYPES:
            rows = [r for r in model_rows if r.get("track") == "Track_B" and r.get("hallucination_type") == hallucination_type]
            parsed = [r for r in rows if r.get("parsed_verdict") in (0, 1) and not r.get("inference_error")]
            errors = [r for r in parsed if int(r["parsed_verdict"]) == 0]
            table2_rows.append({
                "Model": model_name,
                "Prompt_Mode": mode,
                "Hallucination_Type": hallucination_type,
                "Track_B_FNR (%)": 100.0 * len(errors) / len(parsed) if parsed else float("nan"),
                "Collapse_Rate (%)": 100.0 * np.mean([bool(r.get("is_collapsed")) for r in rows]) if rows else float("nan"),
                "Parse_Failure_Rate (%)": 100.0 * np.mean([bool(r.get("parse_failure")) for r in rows]) if rows else float("nan"),
                "N_Rows": len(rows),
                "N_Parsed": len(parsed),
            })
    table2 = pd.DataFrame(table2_rows, columns=[
        "Model", "Prompt_Mode", "Hallucination_Type", "Track_B_FNR (%)", "Collapse_Rate (%)",
        "Parse_Failure_Rate (%)", "N_Rows", "N_Parsed",
    ])
    _write_table(table2, results_dir, "table2_granular_hallucination_types")

    table3_rows = []
    for model_name, mode in model_modes:
        if mode != "cot":
            continue
        for task in TASKS:
            cot = _metric([r for r in records if r["model_name"] == model_name and r["prompt_mode"] == "cot" and r.get("task") == task])["BenHalluScore (%)"]
            zero = _metric([r for r in records if r["model_name"] == model_name and r["prompt_mode"] == "zero_shot" and r.get("task") == task])["BenHalluScore (%)"]
            delta = cot - zero if math.isfinite(cot) and math.isfinite(zero) else float("nan")
            table3_rows.append({
                "Model": model_name,
                "Task": task,
                "BHS_ZeroShot (%)": zero,
                "BHS_CoT (%)": cot,
                "Delta_BHS": delta,
            })
    table3 = pd.DataFrame(table3_rows, columns=["Model", "Task", "BHS_ZeroShot (%)", "BHS_CoT (%)", "Delta_BHS"])
    _write_table(table3, results_dir, "table3_cot_delta_ablation")

    sensitivity_rows = []
    for model_name, mode in model_modes:
        rows = [r for r in records if r["model_name"] == model_name and r["prompt_mode"] == mode]
        parsed = [r for r in rows if r.get("parsed_verdict") in (0, 1) and not r.get("inference_error")]
        a = [int(r["parsed_verdict"]) for r in parsed if r.get("track") == "Track_A"]
        b = [int(r["parsed_verdict"]) for r in parsed if r.get("track") == "Track_B"]
        if not a or not b:
            continue
        for step in range(1, 10):
            w_a = step / 10.0
            w_b = 1.0 - w_a
            metrics = compute_bhs(a, b, w_a=w_a, w_b=w_b)
            sensitivity_rows.append({
                "Model": model_name, "Prompt_Mode": mode, "w_a": w_a, "w_b": w_b,
                "FPR_A (%)": 100 * metrics["FPR_A"], "FNR_B (%)": 100 * metrics["FNR_B"],
                "BenHalluScore (%)": metrics["BHS"],
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    if not sensitivity.empty:
        sensitivity["Rank"] = sensitivity.groupby(["w_a", "w_b"])["BenHalluScore (%)"].rank(method="min", ascending=True)
        rank_ranges = sensitivity.groupby(["Model", "Prompt_Mode"])["Rank"].agg(Rank_Min="min", Rank_Max="max").reset_index()
        sensitivity = sensitivity.merge(rank_ranges, on=["Model", "Prompt_Mode"], how="left")
    _write_table(sensitivity, results_dir, "table4_weight_sensitivity")


def _dialect_table(records: list[dict[str, Any]], results_dir: Path) -> None:
    rows = []
    groups = sorted({(str(r["model_name"]), str(r["prompt_mode"])) for r in records})
    for model, mode in groups:
        current = [r for r in records if r["model_name"] == model and r["prompt_mode"] == mode]
        valid = [r for r in current if r.get("parsed_score") is not None and not r.get("inference_error")]
        ccc = compute_lins_ccc(
            [float(r["human_or_primary_score"]) for r in valid],
            [float(r["parsed_score"]) for r in valid],
        ) if valid else float("nan")
        try:
            cbs = compute_critical_bias_sensitivity(
                [float(r["parsed_score"]) for r in valid], [str(r["dialect"]) for r in valid]
            ) if valid else None
        except ValueError:
            cbs = None
        rows.append({
            "Model": model,
            "Prompt_Mode": mode,
            "Lin_CCC": ccc,
            "Anchor_Variant": cbs["anchor_variant"] if cbs else "NA",
            "Anchor_Mean_Score": cbs["anchor_mean"] if cbs else float("nan"),
            "Chittagong_Mean_Score": cbs["chittagong_mean"] if cbs else float("nan"),
            "CBS_Performance_Drop": cbs["CBS"] if cbs else float("nan"),
            "Collapse_Rate (%)": 100.0 * np.mean([bool(r.get("is_collapsed")) for r in current]) if current else float("nan"),
            "Parse_Failure_Rate (%)": 100.0 * np.mean([bool(r.get("parse_failure")) for r in current]) if current else float("nan"),
            "N_Rows": len(current),
            "N_Scored": len(valid),
        })
    _write_table(pd.DataFrame(rows), results_dir, "table_dialect_ccc_cbs")


def _load_dataset(path: Path) -> tuple[pd.DataFrame, bool]:
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if frame.empty:
        raise ValueError("Input CSV is empty")
    columns = set(frame.columns)
    if BENHALLU_COLUMNS.issubset(columns):
        frame["id"] = frame["id"].astype(str)
        frame["task"] = frame["task"].astype(str)
        frame["track"] = frame["track"].astype(str)
        frame["hallucination_type"] = frame["hallucination_type"].astype(str)
        frame["ground_truth_label"] = pd.to_numeric(frame["ground_truth_label"], errors="raise").astype(int)
        if frame["id"].str.strip().eq("").any():
            raise ValueError("Input split contains an empty id")
        if frame["id"].duplicated().any():
            raise ValueError("Input split contains duplicate ids")
        if not set(frame["task"]).issubset(TASKS):
            raise ValueError(f"Input split contains unknown task values: {sorted(set(frame['task']) - set(TASKS))}")
        if not set(frame["track"]).issubset({"Track_A", "Track_B"}):
            raise ValueError("track values must be Track_A or Track_B")
        if not frame["ground_truth_label"].isin((0, 1)).all():
            raise ValueError("ground_truth_label must be binary")
        expected_labels = frame["track"].map({"Track_A": 0, "Track_B": 1}).astype(int)
        if not frame["ground_truth_label"].equals(expected_labels):
            raise ValueError("ground_truth_label must be 0 for Track_A and 1 for Track_B")
        if not (frame.loc[frame["track"] == "Track_A", "hallucination_type"] == "None").all():
            raise ValueError("Track_A hallucination_type must be 'None'")
        if not frame.loc[frame["track"] == "Track_B", "hallucination_type"].isin(HALLUCINATION_TYPES).all():
            raise ValueError("Track_B hallucination_type must use one of the 12 canonical labels")
        return frame, False
    if DIALECT_COLUMNS.issubset(columns):
        frame["id"] = frame["id"].astype(str)
        frame["dialect"] = frame["dialect"].astype(str)
        frame["question_type"] = frame["question_type"].astype(str)
        frame["human_or_primary_score"] = pd.to_numeric(frame["human_or_primary_score"], errors="raise")
        if not frame["human_or_primary_score"].between(0, 10).all():
            raise ValueError("human_or_primary_score must lie between 0 and 10")
        if frame["id"].str.strip().eq("").any():
            raise ValueError("Input split contains an empty id")
        if frame["id"].duplicated().any():
            raise ValueError("Input split contains duplicate ids")
        if frame["dialect"].str.strip().eq("").any():
            raise ValueError("Input split contains an empty dialect")
        if not set(frame["question_type"]).issubset({"definitional", "factual"}):
            raise ValueError("question_type must be definitional or factual")
        return frame, True
    raise ValueError("Input CSV does not match the BenHalluEval or Dialectal_Bias schema")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True, help="A canonical BenHallu or dialect split CSV")
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+", default=["mock"], help="Model names/IDs; for transformers these are Hugging Face IDs")
    parser.add_argument("--backend", choices=("mock", "ollama", "transformers"), default="mock")
    parser.add_argument("--prompt_modes", nargs="+", choices=PROMPT_MODES, default=["zero_shot", "cot"])
    parser.add_argument("--split_name", help="Checkpoint suffix; defaults to input CSV stem")
    parser.add_argument("--ollama_endpoint", default="http://localhost:11434/api/generate")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if len(set(args.models)) != len(args.models):
            raise ValueError("--models cannot contain duplicate names")
        if len(set(args.prompt_modes)) != len(args.prompt_modes):
            raise ValueError("--prompt_modes cannot contain duplicates")
        frame, dialect_mode = _load_dataset(args.input_csv)
        split_name = _slug(args.split_name or args.input_csv.stem)
        outputs_dir = args.output_dir / "outputs"
        results_dir = args.output_dir / "results"
        for model_name in args.models:
            backend = _make_backend(args.backend, model_name, args.ollama_endpoint, dialect_mode)
            for prompt_mode in args.prompt_modes:
                checkpoint = outputs_dir / f"{_slug(model_name)}_{prompt_mode}_{split_name}.jsonl"
                _infer_records(frame, model_name, prompt_mode, backend, checkpoint, dialect_mode)
                print(f"Checkpoint ready: {checkpoint}")
        records = _read_checkpoint_records(outputs_dir, split_name, set(frame["id"].astype(str)))
        if dialect_mode:
            _dialect_table(records, results_dir)
        else:
            _primary_tables(records, results_dir)
        print(f"Wrote Markdown and CSV summaries to {results_dir.resolve()}")
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, urllib.error.URLError, pd.errors.ParserError) as exc:
        print(f"eval.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
