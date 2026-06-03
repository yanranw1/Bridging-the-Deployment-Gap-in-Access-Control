"""
eval_verifier.py
----------------
Evaluates the trained ACP verifier checkpoint.

Supports both the OLD 12-label checkpoint (labels 0-11, correct=11)
and the NEW 8-label checkpoint (labels 0-7, correct=7).
Set CORRECT_LABEL and ID2AUGS below to match whichever checkpoint you load.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
from dataset.utils import prepare_inputs_bart
from tqdm import tqdm


# ---------------------------------------------------------------------------
# ── Switch this block to match your checkpoint ──────────────────────────────
#
# # OLD checkpoint (12-label, correct=11):
# ID2AUGS = {
#     0: 'allow_deny', 1: 'csub',   2: 'cact',  3: 'cres',
#     4: 'ccond',      5: 'cpur',   6: 'msub',  7: 'mres',
#     8: 'mcond',      9: 'mpur',  10: 'mrules', 11: 'correct',
# }
# CORRECT_LABEL = 11

# NEW checkpoint (8-label, correct=7) — uncomment after retraining:
ID2AUGS = {
    0: 'allow_deny', 1: 'csub', 2: 'cact', 3: 'cres',
    4: 'msub',       5: 'mres', 6: 'mrules', 7: 'correct',
}
CORRECT_LABEL = 7

# ---------------------------------------------------------------------------

VER_MODEL_NAME    = "facebook/bart-large"
verification_ckpt = "../checkpoints/verification/checkpoint-2484/"
ver_device        = "cuda:0"

# Derived — do not edit
AUGS2ID    = {v: k for k, v in ID2AUGS.items()}
ALL_LABELS = sorted(ID2AUGS.keys())
ERROR_NAMES = [ID2AUGS[i] for i in ALL_LABELS]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

ver_test_df   = pd.read_csv("../data/verification/test_verification_dataset.csv")
ver_tokenizer = AutoTokenizer.from_pretrained(VER_MODEL_NAME)
ver_model     = AutoModelForSequenceClassification.from_pretrained(
    verification_ckpt
).to(ver_device)
ver_model.eval()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

truth = ver_test_df["labels"].to_list()
preds = []

for i in tqdm(range(len(ver_test_df))):
    inp        = ver_test_df.iloc[i].inputs
    pp         = ver_test_df.iloc[i].outputs
    ver_inp    = prepare_inputs_bart(inp, pp, ver_tokenizer, ver_device)
    logits     = ver_model(**ver_inp).logits
    pred_class = torch.softmax(logits, dim=1).argmax(dim=-1).item()
    preds.append(pred_class)

# ---------------------------------------------------------------------------
# Binary metrics  (correct=positive, any error=negative)
# ---------------------------------------------------------------------------

def to_binary(labels):
    return [1 if l == CORRECT_LABEL else 0 for l in labels]

truth_bin = to_binary(truth)
preds_bin = to_binary(preds)

print("=== Binary (correct vs. any error) ===")
print(classification_report(truth_bin, preds_bin,
                             target_names=["error", "correct"]))
print(f"MCC: {matthews_corrcoef(np.array(truth_bin), np.array(preds_bin)):.4f}\n")

# ---------------------------------------------------------------------------
# Per-class fine-grained metrics
# Only report classes that actually appear in truth or preds to avoid the
# "Number of classes does not match target_names" error when the checkpoint
# has a different label count than ID2AUGS.
# ---------------------------------------------------------------------------

observed_labels = sorted(set(truth) | set(preds))
observed_names  = [ID2AUGS.get(l, f"LABEL_{l}") for l in observed_labels]

print("=== Per-class (fine-grained) ===")
print(classification_report(truth, preds,
                             labels=observed_labels,
                             target_names=observed_names))