#!/usr/bin/env python3
"""
Fine-tune a seq2seq model (T5/Flan-T5) to convert an ACP (Access Control Policy)
block into a CODE block (api call + args + decision + reason).

Data format (class3.json): a list of records, each with:
  - "NL":   conversation turns (context)
  - "ACP":  list of one access-control-policy dict   (input)
  - "CODE": list of one code dict                     (target)

Usage:
  pip install "transformers>=4.40" datasets torch sentencepiece accelerate
  python train_acp_to_code.py --data class3.json --model google/flan-t5-base --epochs 10

Inference after training:
  python train_acp_to_code.py --predict --model_dir ./acp2code-model --data class3.json
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# --------------------------------------------------------------------------- #
# Serialization: turn ACP / CODE dicts into compact strings the model learns. #
# --------------------------------------------------------------------------- #
def serialize_nl(nl_turns):
    """Flatten conversation context into a single string."""
    parts = []
    for t in nl_turns:
        role = t.get("role", "")
        text = t.get("text", "")
        line = f"{role}: {text}"
        # include any attached resource (e.g. retrieved email) compactly
        if "resource" in t and t["resource"]:
            line += f" [resource: {json.dumps(t['resource'], ensure_ascii=False)}]"
        parts.append(line)
    return " ".join(parts)


def build_input(record):
    """Construct the source text fed to the encoder."""
    nl = serialize_nl(record.get("NL", []))
    acp = record["ACP"][0]  # exactly one ACP block in this dataset
    acp_str = json.dumps(acp, ensure_ascii=False, sort_keys=True)
    return f"convert acp to code | context: {nl} | acp: {acp_str}"


def build_target(record):
    """Construct the target text the decoder must produce (the CODE block)."""
    code = record["CODE"][0]
    # Keep a stable key order so the model learns a consistent template.
    ordered = {
        "acp_id": code.get("acp_id"),
        "api": code.get("api"),
        "args": code.get("args", {}),
        "decision": code.get("decision"),
        "reason": code.get("reason", ""),
    }
    return json.dumps(ordered, ensure_ascii=False)


def load_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [{"input": build_input(r), "target": build_target(r)} for r in raw]


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train(args):
    examples = load_examples(args.data)
    print(f"Loaded {len(examples)} examples.")

    ds = Dataset.from_list(examples)
    split = ds.train_test_split(test_size=args.val_frac, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess(batch):
        model_in = tokenizer(
            batch["input"],
            max_length=args.max_in,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_out,
            truncation=True,
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    tokenized = split.map(preprocess, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_out,
        logging_steps=10,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),

        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print(f"Saved model to {args.model_dir}")


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
def predict(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.eval()

    examples = load_examples(args.data)[: args.n_predict]
    for ex in examples:
        inp = tokenizer(ex["input"], return_tensors="pt", truncation=True,
                        max_length=args.max_in)
        with torch.no_grad():
            out = model.generate(**inp, max_length=args.max_out, num_beams=4)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        print("=" * 80)
        print("INPUT :", ex["input"][:200], "...")
        print("PRED  :", pred)
        print("GOLD  :", ex["target"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_train.json")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--model_dir", default="./acp2code-model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_in", type=int, default=768)
    p.add_argument("--max_out", type=int, default=384)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--predict", action="store_true", help="run inference instead of training")
    p.add_argument("--n_predict", type=int, default=5)
    args = p.parse_args()

    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()