"""
Fine-tune FLAN-T5-base for NL -> CODE generation.

Input  : the NL conversation (User/Agent turns + any retrieved resources)
Output : the full CODE JSON (api, args, decision, reason)

The ACP field is NEVER passed to the model (neither input nor target).
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 512


def serialize_nl(nl_turns):
    """Flatten the NL conversation into a single prompt string.
    Includes retrieved 'resource' blocks (these are things the agent has
    legitimately observed) but NEVER the ACP."""
    lines = []
    for turn in nl_turns:
        role = turn.get("role", "")
        text = turn.get("text", "")
        lines.append(f"{role}: {text}")
        # Agent turns may carry a retrieved resource (email metadata, draft, etc.)
        if "resource" in turn and turn["resource"] is not None:
            res = json.dumps(turn["resource"], ensure_ascii=False)
            lines.append(f"  [retrieved]: {res}")
    convo = "\n".join(lines)
    return (
        "Generate the code action(s) for the following request. "
        "Respond ONLY with a JSON list of code objects, each having "
        "fields: acp_id, api, args, decision, reason.\n\n"
        f"Conversation:\n{convo}\n\nCode:"
    )


def serialize_code(code_list):
    """Target = full CODE JSON (compact, deterministic key order)."""
    return json.dumps(code_list, ensure_ascii=False, separators=(",", ":"))


def strip_policy_metadata(ex):
    """Defensive: ensure ACP policy_metadata never enters the pipeline.
    (NL/CODE are what get serialized; ACP is dropped here entirely.)"""
    for block in ex.get("ACP", []):
        block.pop("policy_metadata", None)
    return ex


def load_examples(path):
    data = json.loads(Path(path).read_text())
    inputs, targets = [], []
    for ex in data:
        ex = strip_policy_metadata(ex)             # blind policy_metadata
        inputs.append(serialize_nl(ex["NL"]))      # ACP excluded
        targets.append(serialize_code(ex["CODE"])) # full CODE JSON
    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_train.json")
    ap.add_argument("--val_file", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.json")
    ap.add_argument("--output_dir", default="outputs/flan_t5_nl2code")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = load_examples(args.train_file)
    val_ds = load_examples(args.val_file) if args.val_file else None

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    if val_ds is not None:
        val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()