"""
NL → ACP Translation Training Script
======================================
Fine-tunes a T5 model to translate natural-language email agent conversations
into ACP (Access Control Policy) JSON objects.

Requirements:
    pip install transformers datasets torch accelerate sentencepiece

Usage:
    python train_nl_to_acp.py --data_path combined_test.csv --output_dir ./nl_acp_model
"""

import argparse
import json
import logging
import os
from typing import Any

import csv
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def format_nl_conversation(nl_turns: list[dict]) -> str:
    """
    Flatten a multi-turn NL conversation into a single prompt string.

    Each turn is prefixed with its role. Agent turns include any retrieved
    resource context so the model can ground its ACP prediction.

    Example output:
        User: Find the email from bookstore@example.com.
        Agent: I found the promotional email from bookstore@example.com.
        [resource] {"id": "msg_1b7d2a8e", ...}
        User: Trash it.
    """
    parts = []
    for turn in nl_turns:
        role = turn["role"]
        text = turn.get("text", "")
        parts.append(f"{role}: {text}")
        if "resource" in turn and turn["resource"]:
            resource_str = json.dumps(turn["resource"], ensure_ascii=False)
            parts.append(f"[resource] {resource_str}")
    return "\n".join(parts)


def format_acp_target(acp_list: list[dict]) -> str:
    """
    Serialize the ACP list to a compact JSON string (the target sequence).
    Using compact JSON keeps the target length down and consistent.
    """
    return json.dumps(acp_list, ensure_ascii=False, separators=(",", ":"))


def load_examples(data_path: str) -> list[dict[str, str]]:
    """Load a CSV file and convert every row to an (input, target) pair.

    Expected CSV columns:
        input   – the NL conversation string (already formatted as plain text)
        output  – the ACP string (JSON or structured text)

    The optional 'acp' column (0/1 label) is ignored during training.
    """
    examples = []
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            input_text = row.get("input", "").strip()
            target_text = row.get("output", "").strip()

            if not input_text or not target_text:
                logger.warning("Skipping row %d — missing input or output.", i)
                continue

            examples.append({
                "input_text": "translate to ACP: " + input_text,
                "target_text": target_text,
            })

    logger.info("Loaded %d examples from %s", len(examples), data_path)
    return examples


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_batch(batch, tokenizer, max_input_len, max_target_len):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_len,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # ── Model & tokenizer ────────────────────────────────────────────────────
    logger.info("Loading model and tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = Dataset.from_list(load_examples(args.data_path))
    eval_ds  = Dataset.from_list(load_examples(args.eval_path))
    logger.info("Train: %d  |  Eval: %d", len(train_ds), len(eval_ds))

    # ── Tokenise ─────────────────────────────────────────────────────────────
    def tokenize_fn(batch: dict) -> dict:
        return tokenize_batch(batch, tokenizer, args.max_input_len, args.max_target_len)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["input_text", "target_text"])
    eval_ds  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=["input_text", "target_text"])

    # ── Collator ─────────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,    # ignore padding in loss
        pad_to_multiple_of=8,
    )

    # ── Training args ────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.05,
        weight_decay=0.01,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=False,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",           # swap to "wandb" / "tensorboard" if desired
        save_total_limit=2,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Starting training…")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved to %s", args.output_dir)


# ---------------------------------------------------------------------------
# Inference helper (run after training)
# ---------------------------------------------------------------------------

def predict(model_dir: str, nl_turns: list[dict]) -> list[dict]:
    """
    Load a saved model and predict the ACP for a new conversation.

    Example:
        turns = [
            {"role": "User", "text": "Find the invoice from billing@acme.com."},
            {"role": "Agent", "text": "Found it.", "resource": {"id": "msg_abc"}},
            {"role": "User", "text": "Archive it."},
        ]
        acp = predict("./nl_acp_model", turns)
        print(acp)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()

    prompt = "translate to ACP: " + format_nl_conversation(nl_turns)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # try:
    #     return json.loads(result)
    # except json.JSONDecodeError:
    #     logger.warning("Model output was not valid JSON:\n%s", result)
    #     return [{"raw_output": result}]
    print("$$:",result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune T5 for NL → ACP translation.")

    # Data
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/agentv-main/email_agent/dataset/combined_train.csv",
                        help="Path to the training CSV file (must have 'input' and 'output' columns)")
    parser.add_argument("--eval_path", type=str, default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.csv",
                        help="Path to the evaluation CSV file (same format as --data_path)")

    # Model
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                        help="HuggingFace model name or local path (default: flan-t5-base). "
                             "Use 't5-large' or 'google/flan-t5-base' for better quality.")
    parser.add_argument("--output_dir", type=str, default="./nl_acp_model-flan-t5",
                        help="Directory to save the fine-tuned model")

    # Sequence lengths
    parser.add_argument("--max_input_len",  type=int, default=512,
                        help="Max tokens for input conversation (default: 512)")
    parser.add_argument("--max_target_len", type=int, default=256,
                        help="Max tokens for target ACP JSON (default: 256)")

    # Training hyper-parameters
    parser.add_argument("--epochs",     type=int,   default=5,    help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,     help="Per-device batch size")
    parser.add_argument("--lr",         type=float, default=3e-4,  help="Learning rate")
    parser.add_argument("--patience",   type=int,   default=3,
                        help="Early-stopping patience in epochs")
    parser.add_argument("--no_fp16",    action="store_true",
                        help="Disable fp16 mixed-precision training")

    args = parser.parse_args()
    main(args)