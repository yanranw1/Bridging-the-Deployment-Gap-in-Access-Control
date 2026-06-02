"""
End-to-end inference pipeline: Identification -> Generation -> Verification
===========================================================================

Flow:
    1. IDENTIFICATION (BERT, 2 classes)
       Reads xxxtest.csv. For each row, predicts whether the NL text is an
       access-control policy (NLACP). Only rows predicted class 1 (True / is
       an NLACP) are passed downstream.

    2. GENERATION (FLAN-T5, seq2seq)
       For each surviving row, generates the ACP string from the NL text using
       train_t5.predict().

    3. VERIFICATION (BART, 12 classes)
       Classifies the (NL input, generated ACP) pair. Only pairs predicted
       class 11 ("correct") are kept as final, verified output.

Usage:
    python run_pipeline.py \
        --test_path xxxtest.csv \
        --id_ckpt        ../checkpoints/identification/email \
        --gen_model_dir  ./nl_acp_model-flan-t5 \
        --ver_ckpt       ../checkpoints/verification/checkpoint/ \
        --output_path    pipeline_output.csv \
        --device cuda:0
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Generation helpers come from the training script (predict + conversation format)
# [location: generation/train_t5.py]
from generation.train_t5 import predict

# Verifier input builder (same helper used in eval_verifier.py)
from verification.dataset.utils import prepare_inputs_bart

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

# Identification: 0 = not an NLACP, 1 = is an NLACP (keep)
ID_KEEP_LABEL = 1

# Verification labels (must match training)
ID2AUGS = {
    0: "allow_deny", 1: "csub", 2: "cact", 3: "cres", 4: "ccond",
    5: "cpur", 6: "msub", 7: "mres", 8: "mcond", 9: "mpur",
    10: "mrules", 11: "correct",
}
VER_CORRECT_LABEL = 11  # only ACPs classified "correct" are kept


# ---------------------------------------------------------------------------
# Stage 1 — Identification
# ---------------------------------------------------------------------------

def load_identifier(ckpt: str, device: str):
    logger.info("Loading identification model from %s", ckpt)
    model = BertForSequenceClassification.from_pretrained(ckpt, num_labels=2).to(device)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(ckpt)
    return model, tokenizer


@torch.no_grad()
def identify(text: str, model, tokenizer, device: str) -> int:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    ).to(device)
    # ACPDataset typically only feeds the model the text fields; drop nothing here
    logits = model(input_ids=enc["input_ids"],
                   attention_mask=enc["attention_mask"]).logits
    return int(torch.softmax(logits, dim=1).argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Stage 3 — Verification
# ---------------------------------------------------------------------------

def load_verifier(ckpt: str, model_name: str, device: str):
    logger.info("Loading verification model from %s", ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.no_grad()
def verify(nl_input: str, acp_pred: str, model, tokenizer, device: str) -> int:
    ver_inp = prepare_inputs_bart(nl_input, acp_pred, tokenizer, device)
    logits = model(**ver_inp).logits
    return int(torch.softmax(logits, dim=1).argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args) -> pd.DataFrame:
    device = args.device

    df = pd.read_csv(args.test_path)
    if args.input_col not in df.columns:
        raise ValueError(
            f"Column '{args.input_col}' not found in {args.test_path}. "
            f"Available columns: {list(df.columns)}"
        )
    logger.info("Loaded %d rows from %s", len(df), args.test_path)

    id_model, id_tok = load_identifier(args.id_ckpt, device)
    ver_model, ver_tok = load_verifier(args.ver_ckpt, args.ver_model_name, device)

    rows = []
    n_identified = 0
    n_verified = 0

    for i, row in df.iterrows():
        nl_text = str(row[args.input_col]).strip()

        # ---- Stage 1: identification ------------------------------------
        id_label = identify(nl_text, id_model, id_tok, device)
        is_nlacp = (id_label == ID_KEEP_LABEL)

        rec = {
            "index": i,
            "input": nl_text,
            "id_label": id_label,
            "is_nlacp": is_nlacp,
            "generated_acp": None,
            "ver_label": None,
            "ver_class": None,
            "verified_correct": False,
        }

        if not is_nlacp:
            # Filtered out at identification — does not proceed.
            rows.append(rec)
            continue
        n_identified += 1

        # ---- Stage 2: generation ----------------------------------------
        nl_turns = [{"role": "User", "text": nl_text}]
        acp_pred = predict(args.gen_model_dir, nl_turns)
        if not isinstance(acp_pred, str):
            acp_pred = str(acp_pred)
        acp_pred = "{"+acp_pred+"}"
        rec["generated_acp"] = acp_pred

        # ---- Stage 3: verification --------------------------------------
        ver_label = verify(nl_text, acp_pred, ver_model, ver_tok, device)
        rec["ver_label"] = ver_label
        rec["ver_class"] = ID2AUGS.get(ver_label, str(ver_label))
        print("ver_label", ver_label)
        rec["verified_correct"] = (ver_label == VER_CORRECT_LABEL)
        if rec["verified_correct"]:
            n_verified += 1

        rows.append(rec)

    out = pd.DataFrame(rows)

    logger.info("=" * 60)
    logger.info("Total rows              : %d", len(df))
    logger.info("Passed identification   : %d", n_identified)
    logger.info("Passed verification     : %d", n_verified)
    logger.info("=" * 60)

    return out


def main():
    p = argparse.ArgumentParser(description="ID -> Generation -> Verification pipeline")
    p.add_argument("--test_path", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.csv",
                   help="Input CSV with the NL text column")
    p.add_argument("--input_col", default="input",
                   help="Name of the column holding the NL conversation text")
    p.add_argument("--id_ckpt", default="/home/ubuntu/agentv-main/checkpoints/identification/email",
                   help="Identification (BERT) checkpoint dir")
    p.add_argument("--gen_model_dir", default="/home/ubuntu/agentv-main/generation/nl_acp_model-flan-t5",
                   help="Generation (T5) model dir")
    p.add_argument("--ver_ckpt", default="/home/ubuntu/agentv-main/checkpoints/verification/checkpoint",
                   help="Verification (BART) checkpoint dir")
    p.add_argument("--ver_model_name", default="facebook/bart-large",
                   help="Base model name for the verifier tokenizer")
    p.add_argument("--output_path", default="pipeline_output.csv")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    out = run_pipeline(args)

    # Full trace of every row
    out.to_csv(args.output_path, index=False)
    logger.info("Full results -> %s", args.output_path)

    # Final accepted output (passed all three stages)
    verified = out[out["verified_correct"]]
    verified_path = args.output_path.replace(".csv", "_verified.csv")
    if verified_path == args.output_path:
        verified_path += "_verified.csv"
    verified[["index", "input", "generated_acp"]].to_csv(verified_path, index=False)
    logger.info("Verified-correct output -> %s (%d rows)", verified_path, len(verified))


if __name__ == "__main__":
    main()