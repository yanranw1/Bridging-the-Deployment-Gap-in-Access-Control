#!/usr/bin/env python3
"""
Evaluate a fine-tuned NL -> CODE seq2seq model (FLAN-T5).

Mirrors train.py exactly:
  - Input  = serialized NL conversation ONLY (ACP is NEVER passed to the model)
  - Target = full CODE JSON list, each object having fields:
             acp_id, api, args, decision, reason

Computes accuracy on:
  - api        : predicted API call name matches gold
  - args       : predicted argument dict matches gold (order-insensitive)
  - decision   : predicted decision matches gold
  - all_three  : api AND args AND decision all correct (strict "logic" metric)

Because both NL->CODE may contain multiple actions, evaluation is done by
aligning predicted code objects to gold code objects by acp_id (falling back
to positional alignment when acp_id is absent).

Usage:
  python test.py \
      --model_dir ./outputs/flan_t5_nl2code \
      --data /home/ubuntu/agentv-main/email_agent/dataset/combined_test.json

Optional:
  --max_in 1024 --max_out 512 --num_beams 4 --batch 8 --dump preds.jsonl
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Reuse the EXACT serialization from training so inputs line up.               #
# ACP is NEVER passed to the model (neither input nor target).                 #
# --------------------------------------------------------------------------- #
def serialize_nl(nl_turns):
    """Flatten the NL conversation into a single prompt string.
    Includes retrieved 'resource' blocks but NEVER the ACP. Identical to
    train.py's serialize_nl."""
    lines = []
    for turn in nl_turns:
        role = turn.get("role", "")
        text = turn.get("text", "")
        lines.append(f"{role}: {text}")
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


def build_input(record):
    """Input is the NL serialization ONLY — ACP is excluded, matching train.py."""
    return serialize_nl(record.get("NL", []))


def gold_code_list(record):
    """Gold = full CODE list, normalized to the evaluated fields."""
    out = []
    for code in record.get("CODE", []):
        out.append({
            "acp_id": code.get("acp_id"),
            "api": code.get("api"),
            "args": code.get("args", {}) or {},
            "decision": code.get("decision"),
        })
    return out


# --------------------------------------------------------------------------- #
# Parsing / comparison helpers                                                 #
# --------------------------------------------------------------------------- #
def parse_prediction(text):
    """
    Parse the model output into a list of code objects.

    The target is a JSON list, so we try, in order:
      1. direct json.loads (expect a list, or a single dict -> wrap)
      2. extract outermost [...] and json.loads
      3. extract outermost {...} (single object) and wrap in a list
      4. brace-repair fallback: the T5 tokenizer drops '{' '}', so the model
         emits a flat key:value stream (with args flattened inline). Split on
         '"acp_id"' boundaries and regex-extract fields per object.
    Returns a list of dicts with keys acp_id/api/args/decision (missing -> None/{}).
    """
    def normalize_obj(obj):
        return {
            "acp_id": obj.get("acp_id"),
            "api": obj.get("api"),
            "args": obj.get("args", {}) or {},
            "decision": obj.get("decision"),
        }

    def to_list(obj):
        if isinstance(obj, list):
            return [normalize_obj(o) for o in obj if isinstance(o, dict)]
        if isinstance(obj, dict):
            return [normalize_obj(obj)]
        return []

    candidates = [text]

    ls, le = text.find("["), text.rfind("]")
    if ls != -1 and le > ls:
        candidates.append(text[ls:le + 1])

    bs, be = text.find("{"), text.rfind("}")
    if bs != -1 and be > bs:
        # wrap the run of objects in a list as a last resort
        candidates.append("[" + text[bs:be + 1] + "]")
        candidates.append(text[bs:be + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            parsed = to_list(obj)
            if parsed:
                return parsed
        except Exception:
            pass

    # --- brace-repair fallback for the flattened, brace-less output -------- #
    return repair_flattened(text)


def _grab_scalar(segment, key):
    """Pull a string/number/bool/null value for `key` from a flat segment."""
    m = re.search(
        rf'"{key}"\s*:\s*("(?:[^"\\]|\\.)*"|null|true|false|-?\d+\.?\d*)',
        segment,
    )
    if not m:
        return None
    raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception:
        return raw.strip('"')


def _grab_args(segment):
    """Args are flattened inline between '"args":' and the next top-level key
    ("decision" or "reason"). Parse the inner key:value pairs into a dict."""
    m = re.search(
        r'"args"\s*:\s*(.*?)\s*,?\s*"(?:decision|reason)"\s*:',
        segment,
        flags=re.DOTALL,
    )
    if not m:
        return {}
    inner = m.group(1).strip().strip("{}").strip()
    args = {}
    for pair in re.finditer(
        r'"([^"]+)"\s*:\s*("(?:[^"\\]|\\.)*"|null|true|false|-?\d+\.?\d*)',
        inner,
    ):
        k, v = pair.group(1), pair.group(2)
        try:
            args[k] = json.loads(v)
        except Exception:
            args[k] = v.strip('"')
    return args


def repair_flattened(text):
    """Split a brace-less, flattened multi-object stream on '"acp_id"'
    boundaries and reconstruct each code object."""
    starts = [m.start() for m in re.finditer(r'"acp_id"\s*:', text)]
    if not starts:
        segments = [text]          # single object without acp_id
    else:
        segments = []
        for idx, s in enumerate(starts):
            e = starts[idx + 1] if idx + 1 < len(starts) else len(text)
            segments.append(text[s:e])

    objs = []
    for seg in segments:
        obj = {
            "acp_id": _grab_scalar(seg, "acp_id"),
            "api": _grab_scalar(seg, "api"),
            "args": _grab_args(seg),
            "decision": _grab_scalar(seg, "decision"),
        }
        if obj["api"] is not None or obj["decision"] is not None:
            objs.append(obj)
    return objs


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


def align(pred_list, gold_list):
    """
    Align predicted objects to gold objects.
    Prefer matching by acp_id; fall back to positional alignment.
    Returns a list of (pred_or_None, gold) pairs, one per gold object.
    """
    pairs = []
    pred_by_id = {}
    for p in pred_list:
        if p.get("acp_id") is not None:
            pred_by_id.setdefault(p["acp_id"], []).append(p)

    used = [False] * len(pred_list)
    for idx, g in enumerate(gold_list):
        gid = g.get("acp_id")
        matched = None
        if gid is not None and pred_by_id.get(gid):
            matched = pred_by_id[gid].pop(0)
            # mark as used
            for j, p in enumerate(pred_list):
                if p is matched and not used[j]:
                    used[j] = True
                    break
        else:
            # positional fallback: next unused prediction
            for j, p in enumerate(pred_list):
                if not used[j]:
                    matched = p
                    used[j] = True
                    break
        pairs.append((matched, g))
    return pairs


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

    for i in range(0, len(records), args.batch):
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
            gold_list = gold_code_list(rec)
            pred_list = parse_prediction(pred_text)

            # Per-record scoring. A field is correct for the record only if the
            # predicted action list matches the gold action list on that field
            # for EVERY action (position-aligned). The empty case is handled
            # naturally: empty gold + empty pred -> all fields correct;
            # a length mismatch -> automatic fail on every field.
            if len(pred_list) != len(gold_list):
                rec_api = rec_args = rec_dec = False
            else:
                rec_api = rec_args = rec_dec = True
                for pred, gold in zip(pred_list, gold_list):
                    if norm(pred.get("api")) != norm(gold["api"]):
                        rec_api = False
                    if not args_equal(pred.get("args", {}) or {}, gold["args"]):
                        rec_args = False
                    if norm(pred.get("decision")) != norm(gold["decision"]):
                        rec_dec = False

            rec_all = rec_api and rec_args and rec_dec

            correct["api"] += rec_api
            correct["args"] += rec_args
            correct["decision"] += rec_dec
            correct["all_three"] += rec_all

            if args.dump:
                dump_rows.append({
                    "input": build_input(rec)[:300],
                    "gold": gold_list,
                    "pred": pred_list,
                    "raw_pred": pred_text,
                    "api_ok": bool(rec_api),
                    "args_ok": bool(rec_args),
                    "decision_ok": bool(rec_dec),
                    "all_three_ok": bool(rec_all),
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
    p.add_argument("--model_dir", default="./outputs/flan_t5_nl2code")
    p.add_argument("--data", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.json")
    p.add_argument("--max_in", type=int, default=1024)
    p.add_argument("--max_out", type=int, default=512)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="evaluate only first N (0 = all)")
    p.add_argument("--dump", default="", help="optional path to write per-example results jsonl")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()