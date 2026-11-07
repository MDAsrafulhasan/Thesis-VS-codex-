from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


WORK = Path("/kaggle/working")
TEST_PATH = Path("/kaggle/input/competitions/bengali-hallucination/test set.csv")
SAMPLE_PATH = Path(
    "/kaggle/input/competitions/bengali-hallucination/dataset samples.json"
)

MODEL_ID = "FacebookAI/xlm-roberta-base"
MODEL_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
DATASET_ID = "SumanMondal/bengali_qa"
DATASET_REVISION = "84e2d15ee0bb19f6e72aabef0dc72bff3609138b"
DATASET_FILE = "data/train-00000-of-00001.parquet"

SEED = 20260719
MAX_LENGTH = 384
TRAIN_PER_LABEL = 20_000
VALID_PER_LABEL = 4_000
EXTERNAL_TARGET_PRECISION = 0.995
EXTERNAL_MIN_COVERAGE = 100
OFFICIAL_MIN_COVERAGE = 5

# The content-independent selection comes from the preceding source-audit
# abstentions. Only context-bearing rows are scored by this relation verifier.
TARGET_IDS = [
    10, 19, 23, 37, 55, 58, 63, 65, 68, 69, 92, 93, 94, 112, 126, 179,
    319, 320, 321, 322, 323, 324, 326, 327, 334, 336, 337, 339, 341, 343,
    344, 345, 346, 347, 349, 350, 352, 353, 354, 355, 356, 358, 359, 362,
    363, 366, 369, 371, 372, 373, 376, 378, 380, 381, 422, 723, 904, 953,
    1029, 1074, 1372, 1540, 1549, 1679, 1680, 1688, 1823, 2158, 2159,
    2160, 2168, 2169, 2174, 2175, 2176, 2182, 2184, 2187, 2188, 2189,
    2190, 2191, 2192, 2193, 2194, 2195, 2196, 2198, 2199, 2200, 2201,
    2203, 2204, 2205, 2206, 2207, 2209, 2210, 2211, 2213, 2214, 2220,
    2268, 2269, 2270, 2272, 2273, 2275, 2276, 2277, 2279, 2280, 2281,
    2283, 2284, 2285, 2286, 2287, 2288, 2290, 2291, 2292, 2293, 2294,
    2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307,
    2308, 2309, 2310, 2311, 2314, 2316, 2317, 2318, 2319, 2341,
]

NUMBER_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")


def canonical(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_context(value: object) -> str:
    text = canonical(value)
    return "" if text.lower() in {"", "[null]", "null", "nan"} else text


def group_key(context: str) -> str:
    return hashlib.sha256(canonical(context).encode()).hexdigest()


def first_answer(value: object) -> str:
    if isinstance(value, dict):
        values = value.get("text")
        if values is not None and len(values) > 0:
            return canonical(values[0])
    return ""


def corrupt_number(answer: str) -> str | None:
    normalized = canonical(answer).translate(
        str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    )
    match = NUMBER_RE.search(normalized)
    if not match:
        return None
    raw = match.group(1)
    number = float(raw)
    if number.is_integer():
        replacement = str(int(number) + (1 if number < 100 else 7))
    else:
        replacement = f"{number + 0.1:g}"
    changed = normalized[: match.start()] + replacement + normalized[match.end() :]
    return changed if changed != normalized else None


def load_source_rows(path: Path) -> dict[str, list[dict]]:
    frame = pd.read_parquet(path, columns=["context", "question", "answers"])
    groups: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in frame.itertuples(index=False):
        context = canonical(row.context)
        question = canonical(row.question)
        answer = first_answer(row.answers)
        key = group_key(context)
        signature = (key, question, answer)
        if not context or not question or not answer or signature in seen:
            continue
        seen.add(signature)
        groups[key].append(
            {"context": context, "prompt_bn": question, "response_bn": answer}
        )
    return {key: rows for key, rows in groups.items() if len(rows) >= 2}


def split_groups(groups: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    keys = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    valid_count = max(1, round(len(keys) * 0.15))
    return keys[valid_count:], keys[:valid_count]


def make_pairs(groups: dict[str, list[dict]], keys: list[str]) -> list[dict]:
    rows = []
    for key in keys:
        records = groups[key]
        for index, item in enumerate(records):
            rows.append(
                {
                    **item,
                    "label": 1,
                    "pair_type": "faithful_gold",
                    "group": key,
                }
            )
            for step in range(1, len(records)):
                decoy = records[(index + step) % len(records)]["response_bn"]
                if canonical(decoy) != canonical(item["response_bn"]):
                    rows.append(
                        {
                            **item,
                            "response_bn": decoy,
                            "label": 0,
                            "pair_type": "same_context_relation_swap",
                            "group": key,
                        }
                    )
                    break
            corrupted = corrupt_number(item["response_bn"])
            if corrupted and canonical(corrupted) not in canonical(item["context"]):
                rows.append(
                    {
                        **item,
                        "response_bn": corrupted,
                        "label": 0,
                        "pair_type": "numeric_relation_corruption",
                        "group": key,
                    }
                )
    return rows


def balanced_cap(rows: list[dict], per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        part = [row for row in rows if row["label"] == label]
        rng.shuffle(part)
        selected.extend(part[:per_label])
    rng.shuffle(selected)
    return selected


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tokenizer(
            "প্রসঙ্গ: "
            + row["context"]
            + "\nপ্রশ্ন: "
            + row["prompt_bn"],
            "উত্তর: " + row["response_bn"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = int(row["label"])
        return encoded


def classification_metrics(eval_prediction) -> dict:
    logits, labels = eval_prediction
    prediction = np.asarray(logits).argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, prediction),
        "macro_f1": f1_score(labels, prediction, average="macro"),
        "f1_0": f1_score(labels, prediction, pos_label=0),
        "f1_1": f1_score(labels, prediction, pos_label=1),
    }


def probabilities(trainer: Trainer, rows: list[dict], tokenizer) -> np.ndarray:
    output = trainer.predict(PairDataset(rows, tokenizer))
    logits = np.asarray(output.predictions)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def choose_threshold(labels: np.ndarray, p0: np.ndarray) -> dict:
    best = None
    for threshold in np.linspace(0.50, 0.999, 500):
        mask = p0 >= threshold
        covered = int(mask.sum())
        if covered < EXTERNAL_MIN_COVERAGE:
            continue
        precision = float((labels[mask] == 0).mean())
        if precision >= EXTERNAL_TARGET_PRECISION:
            best = {
                "threshold": float(threshold),
                "covered": covered,
                "precision": precision,
            }
            break
    return best or {"threshold": None, "covered": 0, "precision": None}


def competition_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.to_dict("records"):
        context = clean_context(row.get("context"))
        if not context:
            continue
        rows.append(
            {
                "id": int(row["id"]),
                "context": context,
                "prompt_bn": canonical(row["prompt_bn"]),
                "response_bn": canonical(row["response_bn"]),
                "label": int(row["label"]) if "label" in row else None,
            }
        )
    return rows


def prediction_frame(rows: list[dict], prob: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [row["id"] for row in rows],
            "p0": prob[:, 0],
            "p1": prob[:, 1],
            "label": [row.get("label") for row in rows],
            "prompt_bn": [row["prompt_bn"] for row in rows],
            "response_bn": [row["response_bn"] for row in rows],
        }
    )


def main() -> None:
    started = time.perf_counter()
    set_seed(SEED)
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    if len(gpu_names) != 2 or not all("T4" in name for name in gpu_names):
        raise RuntimeError(f"expected Tesla T4 x2, observed {gpu_names}")
    for path in (TEST_PATH, SAMPLE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=DATASET_REVISION,
            token=False,
        )
    )
    groups = load_source_rows(source_path)
    train_keys, valid_keys = split_groups(groups)
    train_rows = balanced_cap(
        make_pairs(groups, train_keys), TRAIN_PER_LABEL, SEED
    )
    valid_rows = balanced_cap(
        make_pairs(groups, valid_keys), VALID_PER_LABEL, SEED + 1
    )
    if len(train_rows) < TRAIN_PER_LABEL * 2 or len(valid_rows) < 1_000:
        raise RuntimeError(
            f"insufficient relation data: train={len(train_rows)} valid={len(valid_rows)}"
        )
    print(
        "pairs",
        len(train_rows),
        len(valid_rows),
        Counter(row["pair_type"] for row in train_rows),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=2,
        id2label={0: "hallucinated", 1: "faithful"},
        label2id={"hallucinated": 0, "faithful": 1},
        token=False,
    )
    train_dataset = PairDataset(train_rows, tokenizer)
    valid_dataset = PairDataset(valid_rows, tokenizer)
    scratch = WORK / "relation_training_scratch"
    arguments = TrainingArguments(
        output_dir=str(scratch),
        num_train_epochs=1.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=classification_metrics,
    )
    train_result = trainer.train()
    external_result = trainer.evaluate()
    valid_probability = probabilities(trainer, valid_rows, tokenizer)
    valid_labels = np.asarray([row["label"] for row in valid_rows])
    threshold = choose_threshold(valid_labels, valid_probability[:, 0])

    sample_frame = pd.DataFrame(
        json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    )
    sample_frame.insert(0, "id", range(len(sample_frame)))
    sample_rows = competition_rows(sample_frame)
    sample_probability = probabilities(trainer, sample_rows, tokenizer)
    sample_predictions = prediction_frame(sample_rows, sample_probability)
    sample_predictions.to_csv(
        WORK / "relation_sample_predictions.csv", index=False
    )

    official = {
        "threshold": threshold["threshold"],
        "covered": 0,
        "precision_0": None,
    }
    if threshold["threshold"] is not None:
        mask = sample_probability[:, 0] >= threshold["threshold"]
        official["covered"] = int(mask.sum())
        if mask.any():
            truth = np.asarray([row["label"] for row in sample_rows])
            official["precision_0"] = float((truth[mask] == 0).mean())

    test_frame = pd.read_csv(TEST_PATH)
    test_frame = test_frame[test_frame.id.astype(int).isin(TARGET_IDS)].copy()
    test_rows = competition_rows(test_frame)
    test_probability = probabilities(trainer, test_rows, tokenizer)
    prediction_frame(test_rows, test_probability).to_csv(
        WORK / "relation_test_predictions.csv", index=False
    )

    gate_passed = bool(
        threshold["threshold"] is not None
        and threshold["precision"] is not None
        and threshold["precision"] >= EXTERNAL_TARGET_PRECISION
        and official["covered"] >= OFFICIAL_MIN_COVERAGE
        and official["precision_0"] == 1.0
    )
    checkpoint = None
    if gate_passed:
        checkpoint_path = WORK / "relation_verifier_checkpoint"
        trainer.save_model(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        checkpoint = str(checkpoint_path)
    shutil.rmtree(scratch, ignore_errors=True)

    report = {
        "gpu_names": gpu_names,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_pair_types": dict(
            Counter(row["pair_type"] for row in train_rows)
        ),
        "train_metrics": train_result.metrics,
        "external_metrics": external_result,
        "external_label0_gate": threshold,
        "official_context_label0_gate": official,
        "target_context_rows": len(test_rows),
        "gate_passed": gate_passed,
        "checkpoint": checkpoint,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": (
            "v85 remains default; verifier may propose only label1-to-label0 "
            "relation-swap corrections after external and official precision gates"
        ),
    }
    (WORK / "relation_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
