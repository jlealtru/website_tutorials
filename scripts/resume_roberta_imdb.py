"""Resume RoBERTa+IMDB fine-tuning from the most recent checkpoint under
results/. Used when the nbconvert run is interrupted (terminal closes,
reboot, /clear of a Claude session that owned the parent shell, etc).

Run with: uv run python scripts/resume_roberta_imdb.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# Match the notebook's environment so resume is identical to a fresh run.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(REPO / "notebooks"))
from _utils import pick_device  # also sets PYTORCH_ENABLE_MPS_FALLBACK + HF_HOME

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)


def latest_checkpoint() -> Path | None:
    """Find the largest-numbered checkpoint-N directory under results/."""
    candidates = [
        d for d in RESULTS.glob("checkpoint-*")
        if d.is_dir() and d.name.split("-")[-1].isdigit()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: int(d.name.split("-")[-1]))


def main() -> int:
    device = pick_device()
    print(f"resume: using device={device}")

    ckpt = latest_checkpoint()
    if ckpt is None:
        print(
            f"no checkpoint under {RESULTS}/ yet — run the notebook first, "
            f"this script is for resuming a partial run."
        )
        return 1
    print(f"resume: latest checkpoint = {ckpt.name}")

    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", max_length=512)

    def tokenization(batched_text):
        return tokenizer(batched_text["text"], padding=True, truncation=True)

    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    test_data = test_data.map(tokenization, batched=True, batch_size=len(test_data))
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Load model from checkpoint (preserves classifier head trained so far).
    model = RobertaForSequenceClassification.from_pretrained(str(ckpt))

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    args = TrainingArguments(
        output_dir=str(RESULTS),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=8,
        bf16=True,
        logging_dir=str(RESULTS / "logs"),
        dataloader_num_workers=2,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=False,
        run_name="roberta-classification-resumed",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
    )
    print(f"resume: trainer.train(resume_from_checkpoint='{ckpt}')")
    trainer.train(resume_from_checkpoint=str(ckpt))
    print("resume: trainer.evaluate()")
    print(trainer.evaluate())
    return 0


if __name__ == "__main__":
    sys.exit(main())
