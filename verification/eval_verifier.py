import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import ast
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
from dataset.utils import prepare_inputs_bart, create_out_string, process_label
from tqdm import tqdm


ver_test_df = pd.read_csv("../data/verification/test_verification_dataset.csv")

ID2AUGS = {
    0: "allow_deny",
    1: "csub",
    2: 'cact', 
    3: "cres",
    4: "ccond",
    5: "cpur",
    6: "msub",
    7: "mres",
    8: "mcond",
    9: "mpur",
    10: "mrules",
    11: "correct",
}

AUGS2ID = {k: v for v, k in ID2AUGS.items()}
ERRORS = [k for k, _ in AUGS2ID.items()]

VER_MODEL_NAME = "facebook/bart-large"
verification_ckpt = "../checkpoints/verification/checkpoint/"
ver_device = "cuda:0"


def covert_classes(c):
    return AUGS2ID[c]


truth = ver_test_df["label"].to_list()
truth_bin = []

for i in truth:
    if i == 11:
        truth_bin.append(0)
    else:
        truth_bin.append(1)


preds = []

ver_tokenizer = AutoTokenizer.from_pretrained(VER_MODEL_NAME)

ver_model = AutoModelForSequenceClassification.from_pretrained(verification_ckpt).to(
    ver_device
)
ver_model.eval()

for i in tqdm(range(len(ver_test_df))):

    input = ver_test_df.iloc[i].input
    pp = ver_test_df.iloc[i].policies
    ver_inp = prepare_inputs_bart(input, pp, ver_tokenizer, ver_device)
    pred = ver_model(**ver_inp).logits

    probs = torch.softmax(pred, dim=1)
    pred_class = probs.argmax(axis=-1).item()

    preds.append(pred_class)

preds_bin = []

for i in preds:
    if i == 11:
        preds_bin.append(0)
    else:
        preds_bin.append(1)


print(
    classification_report(truth_bin, preds_bin, target_names=["negative", "positive"])
)
print()
print(f"MCC: {matthews_corrcoef(np.array(truth_bin), np.array(preds_bin))}")
print()
print(classification_report(truth, preds, target_names=ERRORS))
