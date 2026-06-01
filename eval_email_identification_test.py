import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import DataLoader

from identification.classification_dataset import ACPDataset

TEST_PATH = "email_agent/dataset/combined_test_decision.csv"
CKPT_PATH = "checkpoints/identification/email_agent_decision/checkpoint-114"

def main():
    df = pd.read_csv(TEST_PATH)

    print("Test file:", TEST_PATH)
    print("Checkpoint:", CKPT_PATH)
    print("Test shape:", df.shape)

    print("\nACP label counts:")
    print(df["acp"].value_counts())

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(CKPT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("\nDevice:", device)

    ds = ACPDataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    preds = []
    labels = []
    probs_all = []

    with torch.no_grad():
        for batch in loader:
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            probs = torch.softmax(outputs.logits, dim=-1)
            pred_batch = torch.argmax(probs, dim=-1)

            preds.extend(pred_batch.cpu().tolist())
            labels.extend(labels_batch.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

    df["true"] = labels
    df["pred"] = preds
    df["prob_deny_0"] = [p[0] for p in probs_all]
    df["prob_allow_1"] = [p[1] for p in probs_all]

    print("\nPrediction counts:")
    print(df["pred"].value_counts())

    print("\nAccuracy:", accuracy_score(labels, preds))
    print("Macro F1:", f1_score(labels, preds, average="macro"))

    print("\nClassification report:")
    print(classification_report(labels, preds, digits=4, zero_division=0))

    wrong = df[df["true"] != df["pred"]]
    print("\nWrong examples:", len(wrong))
    if len(wrong) > 0:
        cols = [
            "id",
            "class",
            "decision_summary",
            "true",
            "pred",
            "prob_deny_0",
            "prob_allow_1",
            "input",
        ]
        print(wrong[cols].to_string())

if __name__ == "__main__":
    main()

