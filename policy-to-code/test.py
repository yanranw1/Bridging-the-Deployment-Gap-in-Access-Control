#!/usr/bin/env python3
"""
Evaluate a fine-tuned ACP->CODE seq2seq model.

Computes accuracy on three fields:
  - api        : predicted API call name matches gold
  - args       : predicted argument dict matches gold (order-insensitive)
  - decision   : predicted decision matches gold
  - all_three  : api AND args AND decision all correct (the strict / "logic" metric)

Usage:
  python test_acp_to_code.py \
      --model_dir ./acp2code-model \
      --data /home/ubuntu/agentv-main/email_agent/dataset/combined_test.json

Optional:
  --max_in 768 --max_out 384 --num_beams 4 --batch 8 --dump preds.jsonl
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Reuse the EXACT serialization from training so inputs/targets line up.       #
# --------------------------------------------------------------------------- #
def serialize_nl(nl_turns):
    parts = []
    for t in nl_turns:
        role = t.get("role", "")
        text = t.get("text", "")
        line = f"{role}: {text}"
        if "resource" in t and t["resource"]:
            line += f" [resource: {json.dumps(t['resource'], ensure_ascii=False)}]"
        parts.append(line)
    return " ".join(parts)


def build_input(record):
    nl = serialize_nl(record.get("NL", []))
    acp = record["ACP"][0]
    acp_str = json.dumps(acp, ensure_ascii=False, sort_keys=True)
    return f"convert acp to code | context: {nl} | acp: {acp_str}"


def gold_fields(record):
    """Extract the gold api/args/decision straight from the record."""
    code = record["CODE"][0]
    return {
        "api": code.get("api"),
        "args": code.get("args", {}) or {},
        "decision": code.get("decision"),
    }


# --------------------------------------------------------------------------- #
# Parsing / comparison helpers                                                 #
# --------------------------------------------------------------------------- #
def parse_prediction(text):
    """
    Parse the model output into api/args/decision.

    The model often emits a comma-separated key:value stream WITHOUT outer
    braces, and with args flattened, e.g.:
        "api": "create_task", "args": "title": null, "decision": "deny", ...
    So we try, in order:
      1. direct json.loads
      2. wrap in braces and json.loads
      3. extract outermost {...}
      4. regex field extraction (works even on malformed/flattened output)
    """
    candidates = [text]
    stripped = text.strip().rstrip(",")
    candidates.append("{" + stripped + "}")
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return {
                    "api": obj.get("api"),
                    "args": obj.get("args", {}) or {},
                    "decision": obj.get("decision"),
                }
        except Exception:
            pass

    # --- regex fallback: pull fields directly out of the raw text ----------- #
    return regex_extract(text)


def regex_extract(text):
    """Extract api/decision/args from malformed output via regex."""
    def grab_str(key):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        return m.group(1) if m else None

    api = grab_str("api")
    decision = grab_str("decision")

    # args: take everything between "args": and the next top-level field
    # ("decision" or "reason"), then parse the inner key:value pairs.
    args = {}
    m = re.search(
        r'"args"\s*:\s*(.*?)\s*,?\s*"(?:decision|reason)"',
        text,
        flags=re.DOTALL,
    )
    if m:
        inner = m.group(1).strip().strip("{}").strip()
        for pair in re.finditer(
            r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|null|true|false|-?\d+\.?\d*)',
            inner,
        ):
            k, v = pair.group(1), pair.group(2)
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v.strip('"')

    return {"api": api, "args": args, "decision": decision}


def norm(v):
    """Normalize a value for robust comparison (string compare, trimmed)."""
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip()
    return v


def args_equal(a, b):
    """Order-insensitive dict comparison via canonical JSON."""
    try:
        ca = json.dumps(a, sort_keys=True, ensure_ascii=False)
        cb = json.dumps(b, sort_keys=True, ensure_ascii=False)
        return ca == cb
    except Exception:
        return a == b


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    total = len(records)
    correct = {"api": 0, "args": 0, "decision": 0, "all_three": 0}
    dump_rows = []

    for i in range(0, total, args.batch):
        chunk = records[i : i + args.batch]
        inputs = [build_input(r) for r in chunk]
        enc = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=args.max_in,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=args.max_out,
                num_beams=args.num_beams,
            )
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)

        for rec, pred_text in zip(chunk, preds):
            gold = gold_fields(rec)
            pred = parse_prediction(pred_text)

            api_ok = norm(pred["api"]) == norm(gold["api"])
            args_ok = args_equal(pred["args"], gold["args"])
            dec_ok = norm(pred["decision"]) == norm(gold["decision"])
            all_ok = api_ok and args_ok and dec_ok

            correct["api"] += api_ok
            correct["args"] += args_ok
            correct["decision"] += dec_ok
            correct["all_three"] += all_ok

            if args.dump:
                dump_rows.append({
                    "input": build_input(rec)[:300],
                    "gold": gold,
                    "pred": pred,
                    "raw_pred": pred_text,
                    "api_ok": api_ok,
                    "args_ok": args_ok,
                    "decision_ok": dec_ok,
                    "all_three_ok": all_ok,
                })

    print(f"\nEvaluated {total} examples\n" + "-" * 40)
    for k in ["api", "args", "decision", "all_three"]:
        acc = correct[k] / total if total else 0.0
        print(f"{k:<12}: {correct[k]:>4}/{total}  =  {acc:.4f}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote per-example predictions to {args.dump}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./acp2code-model")
    p.add_argument("--data", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.json")
    p.add_argument("--max_in", type=int, default=768)
    p.add_argument("--max_out", type=int, default=384)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="evaluate only first N (0 = all)")
    p.add_argument("--dump", default="", help="optional path to write per-example results jsonl")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()