from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.frame, self.tokenizer, self.max_length = frame.reset_index(drop=True), tokenizer, max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row["context"]), f"প্রশ্ন: {row['prompt_bn']} উত্তর: {row['response_bn']}",
            truncation=True, max_length=self.max_length,
        )
        encoded["labels"] = int(row["label"])
        return encoded


def load_pairs(paths: list[Path], max_rows: int, seed: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_json(path, lines=True)
        if max_rows and len(frame) > max_rows:
            frame = frame.sample(max_rows, random_state=seed)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).fillna("")
    for col in ("context", "prompt_bn", "response_bn", "source", "source_file", "domain"):
        if col not in frame:
            frame[col] = ""
    frame["group"] = frame["source"].where(frame["source"].astype(str).str.len() > 0, frame["source_file"])
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.asarray(logits).argmax(axis=1)
    return {
        "macro_f1": f1_score(labels, pred, average="macro"),
        "f1_0": f1_score(labels, pred, labels=[0], average="macro"),
        "f1_1": f1_score(labels, pred, labels=[1], average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a compact Bengali/multilingual NLI cross-encoder.")
    parser.add_argument("--train-file", type=Path, action="append", required=True)
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2_encoder"))
    parser.add_argument("--max-rows-per-file", type=int, default=30_000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_pairs(args.train_file, args.max_rows_per_file, args.seed)
    split = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=args.seed)
    train_idx, valid_idx = next(split.split(frame, frame.label, frame.group))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    train, valid = PairDataset(frame.iloc[train_idx], tokenizer, args.max_length), PairDataset(frame.iloc[valid_idx], tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, fp16=torch.cuda.is_available(),
        report_to="none", seed=args.seed, save_total_limit=2,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=valid,
                      processing_class=tokenizer, compute_metrics=metrics)
    trainer.train()
    result = trainer.evaluate()
    trainer.save_model(args.output_dir / "best")
    tokenizer.save_pretrained(args.output_dir / "best")
    metadata = {
        "model": args.model, "train_rows": len(train), "validation_rows": len(valid),
        "held_out_groups": sorted(frame.iloc[valid_idx].group.astype(str).unique()), "metrics": result,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
