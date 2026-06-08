
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, matthews_corrcoef
from dataset.utils import prepare_inputs_bart
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ── Configuration ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
INPUT_CSV_PATH  = "/home/ubuntu/agentv-main/generation/results.csv"
OUTPUT_CSV_PATH = "./inference_results.csv"

# NEW checkpoint configuration (8-label, correct=7)
ID2AUGS = {
    0: 'allow_deny', 1: 'csub', 2: 'cact', 3: 'cres',
    4: 'msub',       5: 'mres', 6: 'mrules', 7: 'correct',
}
CORRECT_LABEL = 7

VER_MODEL_NAME    = "facebook/bart-large"
verification_ckpt = "../checkpoints/verification/checkpoint-2484/"
ver_device        = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# ── Load Model & Data ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
print(f"Loading model from checkpoint: {verification_ckpt}...")
ver_tokenizer = AutoTokenizer.from_pretrained(VER_MODEL_NAME)
ver_model     = AutoModelForSequenceClassification.from_pretrained(
    verification_ckpt
).to(ver_device)
ver_model.eval()

print(f"Reading dataset from: {INPUT_CSV_PATH}...")
df = pd.read_csv(INPUT_CSV_PATH)

# ---------------------------------------------------------------------------
# ── Inference & Evaluation Loop ────────────────────────────────────────────
# ---------------------------------------------------------------------------
decisions = []
error_types = []
preds_bin = []
truth_bin = []

print("Running inference and calculating verification performance...")
with torch.no_grad():
    for i in tqdm(range(len(df))):
        # Mapping to your new CSV columns
        inp = df.iloc[i]['input']
        pp  =f"{{df.iloc[i]['predicted']}}"  # The model generation being verified
        exp = df.iloc[i]['expected']   # The gold standard text reference
        
        # 1. Run Verifier Model
        ver_inp    = prepare_inputs_bart(inp, pp, ver_tokenizer, ver_device)
        logits     = ver_model(**ver_inp).logits
        pred_class = torch.softmax(logits, dim=1).argmax(dim=-1).item()
        
        # 2. Track Verifier Predictions
        if pred_class == CORRECT_LABEL:
            decisions.append("accept")
            error_types.append("none")
            preds_bin.append(1)  # 1 = accepted/correct
        else:
            decisions.append("deny")
            error_types.append(ID2AUGS.get(pred_class, f"unknown_error_{pred_class}"))
            preds_bin.append(0)  # 0 = denied/error
            
        # 3. Determine Ground Truth Verification Status
        # Clean string spaces to avoid false mismatches
        is_correct_truth = 1 if str(pp).strip() == str(exp).strip() else 0
        truth_bin.append(is_correct_truth)

# Append tracking metrics back to the original dataframe structure
df["verifier_decision"] = decisions
df["verifier_error_type"] = error_types

# ---------------------------------------------------------------------------
# ── Save Output ────────────────────────────────────────────────____________
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nSuccessfully saved prediction results to: {OUTPUT_CSV_PATH}\n")

# ---------------------------------------------------------------------------
# ── Verification Performance Metrics ───────────────────────────────────────
# ---------------------------------------------------------------------------
print("=== Verification Performance: Binary (Accept vs. Deny) ===")
print(classification_report(
    truth_bin, 
    preds_bin, 
    target_names=["deny (error)", "accept (correct)"]
))
print(f"MCC: {matthews_corrcoef(np.array(truth_bin), np.array(preds_bin)):.4f}\n")