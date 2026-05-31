"""
Fine-tune a seq2seq model (Flan-T5) to translate
natural language conversations → ACP policy strings.

Dataset columns expected:
  - input  : NL conversation (Llama-style chat format)
  - output : ACP decision string

Usage:
    pip install transformers datasets torch pandas scikit-learn
    python train_nl_to_acp.py

    # To run inference after training:
    python train_nl_to_acp.py --infer
"""

import argparse
import os
import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH   = "/home/ubuntu/agentv-main/email_agent/dataset/combined_test.csv"      # path to your CSV
MODEL_NAME  = "google/flan-t5-base"    # swap for flan-t5-large if you have GPU RAM
OUTPUT_DIR  = "./acp_model"
MAX_INPUT   = 512                      # token limit for NL input
MAX_TARGET  = 256                      # token limit for ACP output
BATCH_SIZE  = 4
EPOCHS      = 20
LR          = 5e-4
WARMUP_STEPS = 50
VAL_SIZE    = 0.15                     # 15 % held out for validation
SEED        = 42

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def clean_input(raw: str) -> str:
    """Strip Llama chat tokens; keep only the text content."""
    text = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", "", raw)
    text = re.sub(r"<\|eot_id\|>", "\n", text)
    return text.strip()


def build_prompt(nl_text: str) -> str:
    """Wrap the NL conversation in a task prefix for the model."""
    return (
        "Translate the following agent conversation into an ACP policy decision:\n\n"
        + nl_text
    )


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ACPDataset(Dataset):
    def __init__(self, inputs: list[str], targets: list[str], tokenizer, max_in, max_tgt):
        self.inputs   = inputs
        self.targets  = targets
        self.tokenizer = tokenizer
        self.max_in   = max_in
        self.max_tgt  = max_tgt

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        model_inputs = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_in,
            truncation=True,
            padding=False,
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                self.targets[idx],
                max_length=self.max_tgt,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train():
    # --- Load data ---
    df = pd.read_csv(DATA_PATH)
    assert "input" in df.columns and "output" in df.columns, (
        "CSV must have 'input' and 'output' columns."
    )

    prompts = [build_prompt(clean_input(t)) for t in df["input"].tolist()]
    targets = df["output"].tolist()

    train_x, val_x, train_y, val_y = train_test_split(
        prompts, targets, test_size=VAL_SIZE, random_state=SEED
    )

    print(f"Train: {len(train_x)}  |  Val: {len(val_x)}")

    # --- Tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # --- Datasets ---
    train_ds = ACPDataset(train_x, train_y, tokenizer, MAX_INPUT, MAX_TARGET)
    val_ds   = ACPDataset(val_x,   val_y,   tokenizer, MAX_INPUT, MAX_TARGET)

    # --- Collator handles padding dynamically ---
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # --- Training args ---
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="none",        # set to "wandb" if you want W&B logging
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Starting training …")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def infer(nl_conversation: str) -> str:
    """Load the fine-tuned model and generate an ACP string."""
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    model.eval()

    prompt = build_prompt(clean_input(nl_conversation))
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run interactive inference instead of training.",
    )
    args = parser.parse_args()

    if args.infer:
        if not os.path.isdir(OUTPUT_DIR):
            raise FileNotFoundError(
                f"No trained model found at '{OUTPUT_DIR}'. Run training first."
            )
        print("=== ACP Inference Mode ===  (Ctrl-C to quit)\n")
        while True:
            nl = input("Enter NL conversation:\n> ")
            if not nl.strip():
                continue
            acp = infer(nl)
            print(f"\nACP output:\n{acp}\n")
    else:
        train()