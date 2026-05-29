"""
NL → ACP Translation — Evaluation Script
==========================================
Evaluates a fine-tuned T5 model on a held-out test set (or any JSON file
with the same schema as train.json).

Metrics reported
----------------
- Exact match     : ACP JSON is identical after normalisation
- Key-field match : decision / action / subject / acp_id all correct
- Valid JSON rate  : fraction of outputs that parse as valid JSON
- BLEU            : token-level sequence similarity (sacrebleu)

Requirements:
    pip install transformers torch sacrebleu tqdm

Usage:
    # Basic — point at your saved model and test file
    python test_nl_to_acp.py --model_dir ./nl_acp_model --data_path test.json

    # Use a subset of train.json as a quick sanity check
    python test_nl_to_acp.py --model_dir ./nl_acp_model --data_path train.json --max_samples 50

    # Save per-example predictions to a JSON file for further inspection
    python test_nl_to_acp.py --model_dir ./nl_acp_model --data_path test.json --output_file results.json
"""

import argparse
import json
import logging
import sys
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not installed — BLEU score will be skipped. "
          "Run: pip install sacrebleu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Key ACP fields used for partial / field-level matching
KEY_FIELDS = ["acp_id", "decision", "action", "subject"]

import re
import json

def fix_malformed_acp_string(raw_output: str) -> str:
    """
    Parses the flat, text-flattened T5 output sequence and reconstructs 
    the proper nested JSON structure containing lists of policy dictionaries.
    """
    # 1. Strip whitespace and outer raw brackets if present
    text = raw_output.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    

    lst = re.split(r"[,:]", text)
    result = {}
    policy_metadata = {}
    def find_match(keyword,dictionary):
        for i, ele in enumerate(lst):
            if keyword in ele:
                val = lst[i+1]
                val = val.replace('"', '').replace("'", "")
                val = val.lower()

                if val == "null":
                    val = None
                elif val == "true":
                    val = True
                elif val == "false":
                    val = False
                dictionary[keyword] = val
                lst[i]=""
                lst[i+1]=""
                break


    regular_key =  [
                "acp_id",
                "decision",
                "subject",
                "action",
                "purpose",
                "condition",
    ]

    policy_metadata_key = ["contains_sensitive_information","is_sensitive_action","has_conflicting_constraints","contains_ambiguity"]
    special_key = {"resource"}
    for key in regular_key:
        find_match(key,result)
    
    for key in policy_metadata_key:
        find_match(key,policy_metadata)
    result["policy_metadata"] = policy_metadata
    resource = {}
    for i, ele in enumerate(lst):
        if "policy_metadata" in ele:
            lst[i] = ""
        elif "resource" in ele:
            lst[i] = ""
    lst = [x for x in lst if x != ""]
    if len(lst)%2: #n
        if len(lst) == 1 and "null" in lst[0].lower():
            resource =None
            return result.copy()
        else:
            print(lst)
            print("$$error",raw_output)
            return{}
    for i in range(0,len(lst),2):
        resource[lst[i]] = lst[i+1]
    
    result["resource"]=resource
    # print(raw_output,result)
    return result.copy()

    # # 2. Tokenize into individual key-value expressions.
    # # This splits by commas, but ignores commas that live inside quoted strings.
    # pattern = r'(?:"[^"]*"|[^,]+)'
    # pairs = re.findall(pattern, text)
    
    # policies = []
    # current_policy = None
    # in_metadata = False

    # for pair in pairs:
    #     pair = pair.strip()
    #     if not pair:
    #         continue
            
    #     # Split on the first colon to isolate the key and the value
    #     if ":" not in pair:
    #         continue
    #     key, val = pair.split(":", 1)
        
    #     # Clean quotes and whitespace from the extracted keys/values
    #     key = key.strip().strip('"')
    #     val = val.strip()
        
    #     # Parse standard scalar types (null, booleans) safely
    #     if val.lower() == "null":
    #         parsed_val = None
    #     elif val.lower() == "true":
    #         parsed_val = True
    #     elif val.lower() == "false":
    #         parsed_val = False
    #     else:
    #         parsed_val = val.strip('"')

    #     # Whenever we encounter an 'acp_id', a brand new object block has started
    #     if key == "acp_id":
    #         if current_policy is not None:
    #             policies.append(current_policy)
    #         current_policy = {
    #             "acp_id": parsed_val,
    #             "decision": None,
    #             "subject": None,
    #             "action": None,
    #             "resource": None,
    #             "purpose": None,
    #             "condition": None,
    #             "policy_metadata": {}
    #         }
    #         in_metadata = False
    #         continue

    #     if current_policy is not None:
    #         if key == "policy_metadata":
    #             # The model outputted "policy_metadata": "contains_sensitive_information": true
    #             # We skip setting it directly, and route the next keys into the metadata sub-object
    #             in_metadata = True
                
    #             # Double-check if the value itself contains the first nested key due to splitting flaws
    #             if ":" in val:
    #                 sub_key, sub_val = val.split(":", 1)
    #                 sub_key = sub_key.strip().strip('"')
    #                 sub_val = sub_val.strip()
    #                 if sub_val.lower() == "true": sub_parsed = True
    #                 elif sub_val.lower() == "false": sub_parsed = False
    #                 else: sub_parsed = sub_val.strip('"')
    #                 current_policy["policy_metadata"][sub_key] = sub_parsed
    #         elif in_metadata:
    #             # Assign metadata keys to our nested dictionary structure
    #             current_policy["policy_metadata"][key] = parsed_val
    #         else:
    #             # Assign standard global policy attributes
    #             current_policy[key] = parsed_val

    # # Append the last active object parsed from the sequence
    # if current_policy is not None:
    #     policies.append(current_policy)

    # # Return clean, beautifully formatted JSON
    # print("policies",policies)
    return json.dumps(result, indent=2, ensure_ascii=False)
    
# ---------------------------------------------------------------------------
# Data helpers (mirrors train_nl_to_acp.py)
# ---------------------------------------------------------------------------

def format_nl_conversation(nl_turns: list[dict]) -> str:
    parts = []
    for turn in nl_turns:
        role = turn["role"]
        text = turn.get("text", "")
        parts.append(f"{role}: {text}")
        if "resource" in turn and turn["resource"]:
            resource_str = json.dumps(turn["resource"], ensure_ascii=False)
            parts.append(f"[resource] {resource_str}")
    return "\n".join(parts)


def load_test_data(data_path: str, max_samples: int | None = None) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if max_samples:
        raw = raw[:max_samples]

    examples = []
    for item in raw:
        nl_turns = item.get("NL", [])
        acp = item.get("ACP", [])
        if not nl_turns or not acp:
            continue
        examples.append({
            "id": item.get("id", "unknown"),
            "class": item.get("class"),
            "input_text": "translate to ACP: " + format_nl_conversation(nl_turns),
            "target": acp,
            "target_str": json.dumps(acp, ensure_ascii=False, separators=(",", ":")),
        })

    logger.info("Loaded %d test examples", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Normalisation & matching
# ---------------------------------------------------------------------------

def normalise_acp(acp: list[dict]) -> list[dict]:
    """
    Sort ACP list by acp_id and sort dict keys so that two semantically
    identical ACPs always compare equal regardless of ordering.
    """
    normalised = []
    for entry in acp:
        normalised.append(dict(sorted(entry.items())))
    return sorted(normalised, key=lambda x: x.get("acp_id", ""))


def is_exact_match(pred: list[dict], target: list[dict]) -> bool:
    return normalise_acp(pred) == normalise_acp(target)


def key_field_match(pred: list[dict], target: list[dict]) -> bool:
    """
    True if every ACP entry in the prediction has all KEY_FIELDS matching
    the corresponding target entry (same length required).
    """
    if len(pred) != len(target):
        return False
    pred_s  = sorted(pred,   key=lambda x: x.get("acp_id", ""))
    tgt_s   = sorted(target, key=lambda x: x.get("acp_id", ""))
    for p, t in zip(pred_s, tgt_s):
        for field in KEY_FIELDS:
            if p.get(field) != t.get(field):
                return False
    return True


def per_field_accuracy(predictions: list[list[dict]], targets: list[list[dict]]) -> dict[str, float]:
    """Per-field accuracy across the dataset (only where both pred and target have the field)."""
    counts: dict[str, int] = {f: 0 for f in KEY_FIELDS}
    totals: dict[str, int] = {f: 0 for f in KEY_FIELDS}

    for pred_list, tgt_list in zip(predictions, targets):
        if len(pred_list) != len(tgt_list):
            continue
        pred_s = sorted(pred_list, key=lambda x: x.get("acp_id", ""))
        tgt_s  = sorted(tgt_list,  key=lambda x: x.get("acp_id", ""))
        for p, t in zip(pred_s, tgt_s):
            for field in KEY_FIELDS:
                if field in t:
                    totals[field] += 1
                    if p.get(field) == t.get(field):
                        counts[field] += 1

    return {
        f: (counts[f] / totals[f] * 100 if totals[f] else 0.0)
        for f in KEY_FIELDS
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def batch_predict(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    input_texts: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    num_beams: int,
) -> list[str]:
    """Run batched generation and return raw string outputs."""
    all_outputs = []
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Generating"):
        batch = input_texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_outputs.extend(decoded)
    return all_outputs


def try_parse_json(text: str) -> tuple[bool, Any]:
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            parsed = [parsed]
        return True, parsed
    except json.JSONDecodeError:
        return False, None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(metrics: dict, per_field: dict[str, float]) -> None:
    width = 50
    print("\n" + "=" * width)
    print("  NL → ACP Evaluation Results")
    print("=" * width)
    print(f"  Samples evaluated  : {metrics['n_samples']}")
    print(f"  Valid JSON rate     : {metrics['valid_json_pct']:.1f}%")
    print(f"  Exact match        : {metrics['exact_match_pct']:.1f}%")
    print(f"  Key-field match    : {metrics['key_field_match_pct']:.1f}%")
    if "bleu" in metrics:
        print(f"  BLEU               : {metrics['bleu']:.2f}")
    print("-" * width)
    print("  Per-field accuracy (on length-matched pairs):")
    for field, acc in per_field.items():
        print(f"    {field:<20}: {acc:.1f}%")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load model
    logger.info("Loading model from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load data
    examples = load_test_data(args.data_path, args.max_samples)
    if not examples:
        logger.error("No valid examples found. Exiting.")
        sys.exit(1)

    input_texts  = [e["input_text"]  for e in examples]
    targets      = [e["target"]      for e in examples]
    target_strs  = [e["target_str"]  for e in examples]

    # Generate predictions
    raw_preds = batch_predict(
        model, tokenizer, input_texts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        num_beams=args.num_beams,
    )
    for raw_pred in raw_preds:
        count = raw_pred.count("acp_i")
        res = []
        if count > 1:
            # print("111",count)
            prev = 0
            for i in range(2,1+count):
                # print("here",f'"acp_id":"acp_{i}"')
                end = raw_pred.find(f'"acp_id":"acp_{i}"')
                # print(prev,end)
                res.append(fix_malformed_acp_string(raw_pred[prev:end]))
                prev = end
            res.append(fix_malformed_acp_string(raw_pred[prev:end]))
            
        else:
            res.append(fix_malformed_acp_string(raw_pred))
        print(res)

    # Parse & evaluate
    # valid_json_count  = 0
    # exact_match_count = 0
    # key_field_count   = 0
    # parsed_preds      = []
    # records           = []

    # for ex, raw, tgt, tgt_str in zip(examples, raw_preds, targets, target_strs):
    #     ok, pred_acp = try_parse_json(raw)
    #     valid_json_count  += int(ok)
    #     exact = key_f = False

    #     if ok:
    #         exact = is_exact_match(pred_acp, tgt)
    #         key_f = key_field_match(pred_acp, tgt)
    #     else:
    #         pred_acp = []

    #     exact_match_count += int(exact)
    #     key_field_count   += int(key_f)
    #     parsed_preds.append(pred_acp)

    #     records.append({
    #         "id":              ex["id"],
    #         "class":           ex["class"],
    #         "input":           ex["input_text"],
    #         "target":          tgt,
    #         "prediction_raw":  raw,
    #         "prediction":      pred_acp,
    #         "valid_json":      ok,
    #         "exact_match":     exact,
    #         "key_field_match": key_f,
    #     })

    # n = len(examples)
    # metrics: dict[str, Any] = {
    #     "n_samples":           n,
    #     "valid_json_pct":      valid_json_count  / n * 100,
    #     "exact_match_pct":     exact_match_count / n * 100,
    #     "key_field_match_pct": key_field_count   / n * 100,
    # }

    # # BLEU (token-level, treating JSON strings as sentences)
    # if HAS_SACREBLEU:
    #     bleu = sacrebleu.corpus_bleu(raw_preds, [target_strs])
    #     metrics["bleu"] = bleu.score

    # per_field = per_field_accuracy(parsed_preds, targets)
    # print_results(metrics, per_field)

    # # Optional: break down exact match by class
    # class_stats: dict[int, dict] = {}
    # for rec in records:
    #     cls = rec["class"]
    #     if cls not in class_stats:
    #         class_stats[cls] = {"total": 0, "exact": 0, "key_field": 0}
    #     class_stats[cls]["total"]     += 1
    #     class_stats[cls]["exact"]     += int(rec["exact_match"])
    #     class_stats[cls]["key_field"] += int(rec["key_field_match"])

    # print("  Breakdown by class:")
    # for cls in sorted(class_stats):
    #     s = class_stats[cls]
    #     em  = s["exact"]     / s["total"] * 100
    #     kfm = s["key_field"] / s["total"] * 100
    #     print(f"    class {cls}: {s['total']:>4} samples | exact {em:5.1f}% | key-field {kfm:5.1f}%")
    # print()

    # # Save detailed results
    # if args.output_file:
    #     with open(args.output_file, "w", encoding="utf-8") as f:
    #         json.dump({"metrics": metrics, "per_field": per_field, "records": records}, f, indent=2)
    #     logger.info("Detailed results saved to %s", args.output_file)

    # # Print a few failure examples for quick inspection
    # if args.show_failures > 0:
    #     failures = [r for r in records if not r["exact_match"]][:args.show_failures]
    #     print(f"--- First {len(failures)} failure(s) ---")
    #     for r in failures:
    #         print(f"\nID: {r['id']}  class: {r['class']}")
    #         print(f"  TARGET : {json.dumps(r['target'], separators=(',', ':'))}")
    #         print(f"  PRED   : {r['prediction_raw']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned NL → ACP model.")

    parser.add_argument("--model_dir",    type=str, default="/home/ubuntu/agentv-main/generation/nl_acp_model",
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--data_path",    type=str, default="/home/ubuntu/agentv-main/email_agent/dataset/val.json",
                        help="Path to the test JSON file (same schema as train.json)")
    parser.add_argument("--max_samples",  type=int, default=None,
                        help="Limit evaluation to first N examples (useful for quick checks)")
    parser.add_argument("--batch_size",   type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per prediction (default: 256)")
    parser.add_argument("--num_beams",    type=int, default=4,
                        help="Beam search width (default: 4; use 1 for greedy)")
    parser.add_argument("--output_file",  type=str, default=None,
                        help="Optional path to save full per-example results as JSON")
    parser.add_argument("--show_failures", type=int, default=5,
                        help="Print this many failure examples to stdout (default: 5)")

    args = parser.parse_args()
    main(args)