"""
NL → ACP Inference Script
==========================
Reads a JSON file in the same format as train.json, runs the model on every
entry, prints a side-by-side comparison of predicted vs. ground-truth ACP,
compares decision & action fields, and reports accuracy at the end.

Usage:
    python inference_nl_to_acp.py --input_json test.json
    python inference_nl_to_acp.py --input_json test.json --output_json predictions.json
"""

import argparse
import json
import logging
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Known ACP top-level fields and policy_metadata sub-fields
_TOP_LEVEL_FIELDS = {
    "acp_id", "decision", "subject", "action",
    "purpose", "condition", "resource",
}
_POLICY_META_FIELDS = {
    "contains_sensitive_information", "is_sensitive_action",
    "has_conflicting_constraints", "contains_ambiguity",
}


# ---------------------------------------------------------------------------
# Formatting  (must match train_nl_to_acp.py exactly)
# ---------------------------------------------------------------------------

def format_nl_conversation(nl_turns: list[dict]) -> str:
    parts = []
    for turn in nl_turns:
        role = turn.get("role", "User")
        text = turn.get("text", "")
        parts.append(f"{role}: {text}")
        if "resource" in turn and turn["resource"]:
            resource_str = json.dumps(turn["resource"], ensure_ascii=False)
            parts.append(f"[resource] {resource_str}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

def _try_parse(s: str) -> "list[dict] | None":
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return None


def _fix_nested_dict_fields(s: str) -> str:
    """
    Fix dict-valued fields (e.g. policy_metadata) whose {} were dropped:
        "policy_metadata":"key":true,...  →  "policy_metadata":{"key":true,...}
    """
    for field in ["policy_metadata"]:
        pattern = rf'("{re.escape(field)}"):"'
        match = re.search(pattern, s)
        if match:
            start = match.end() - 1
            outer_end = s.rfind("]")
            if outer_end == -1:
                outer_end = s.rfind("}")
            if outer_end > start:
                nested_content = s[start:outer_end]
                s = s[:match.end() - 1] + "{" + nested_content + "}" + s[outer_end:]
    return s


def _normalize_policy(raw: str) -> "list[dict] | None":
    """
    User-proven fallback: string-patch the flat model output into valid JSON
    then redistribute keys into top-level / resource / policy_metadata buckets.

    Handles output like:
        ["acp_id":"acp_1","decision":"deny",...,"policy_metadata":"key":true,...]
    """
    try:
        s = raw
        # Patch resource value: "resource":"..." → "resource":{"..."
        s = s.replace('"resource":"', '"resource":{"')
        s = s.replace(',"purpose"', '},"purpose"')
        # Patch policy_metadata value
        s = s.replace('"policy_metadata":', '"policy_metadata":{')
        # Replace outer [] with {} (model wrapped object in array brackets)
        s = s.replace('[', '{', 1)
        s = s.replace(']', '}}', 1)

        data = json.loads(s)

        result = {}
        resource = {}
        policy_metadata = {}

        for key, value in data.items():
            if key in _TOP_LEVEL_FIELDS:
                result[key] = value
            elif key in _POLICY_META_FIELDS:
                policy_metadata[key] = value
            else:
                resource[key] = value

        # Only override resource/policy_metadata if they weren't already parsed
        if not isinstance(result.get("resource"), dict):
            result["resource"] = resource
        if not isinstance(result.get("policy_metadata"), dict):
            result["policy_metadata"] = policy_metadata

        return [result]

    except (json.JSONDecodeError, Exception):
        return None


def repair_and_parse(raw: str) -> list[dict]:
    """
    Repair ladder for malformed model output:
      1. Parse as-is.
      2. Fix dropped {} around nested dict fields (policy_metadata).
      3. Fix unquoted Python booleans / None.
      4. Wrap bare array content in {}.
      5. Add missing outer [].
      6. normalize_policy() string-patching fallback.
      7. Store raw string.
    """
    # Pass 1 — as-is
    result = _try_parse(raw)
    if result is not None:
        return result

    repaired = raw.strip()

    # Pass 2 — fix nested dict-valued fields first
    repaired = _fix_nested_dict_fields(repaired)

    # Pass 3 — fix unquoted Python literals
    repaired = re.sub(r':\s*True\b',  ': true',  repaired)
    repaired = re.sub(r':\s*False\b', ': false', repaired)
    repaired = re.sub(r':\s*None\b',  ': null',  repaired)

    # Pass 4 — wrap bare array content in {}
    if repaired.startswith("[") and repaired.endswith("]"):
        inner = repaired[1:-1].strip()
        if inner and not inner.lstrip().startswith("{"):
            candidate = "[{" + inner + "}]"
            result = _try_parse(candidate)
            if result is not None:
                logger.info("JSON repaired: wrapped bare object in {}.")
                return result

    result = _try_parse(repaired)
    if result is not None:
        logger.info("JSON repaired: fixed nested dicts / booleans.")
        return result

    # Pass 5 — missing outer []
    if repaired.startswith("{"):
        result = _try_parse("[" + repaired + "]")
        if result is not None:
            logger.info("JSON repaired: added missing outer [].")
            return result

    # Pass 6 — normalize_policy string-patch fallback
    result = _normalize_policy(raw)
    if result is not None:
        logger.info("JSON repaired: normalize_policy fallback succeeded.")
        return result

    # Pass 7 — give up
    logger.warning("Could not repair JSON output — storing raw string:\n%s", raw)
    print("$$raw",raw)
    return [{"raw_output": raw}]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class ACPPredictor:
    def __init__(
        self,
        model_dir: str = "./nl_acp_model",
        device: str | None = None,
        max_input_len: int = 512,
        max_new_tokens: int = 512,
        num_beams: int = 4,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_input_len  = max_input_len
        self.max_new_tokens = max_new_tokens
        self.num_beams      = num_beams

        logger.info("Loading model from '%s' on %s …", model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        logger.info("Model ready.\n")

    def predict_batch(self, conversations: list[list[dict]]) -> list[list[dict]]:
        prompts = [
            "translate to ACP: " + format_nl_conversation(turns)
            for turns in conversations
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self.max_input_len,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        results = []
        for ids in output_ids:
            raw = self.tokenizer.decode(ids, skip_special_tokens=True)
            results.append(repair_and_parse(raw))
        return results


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def get_field(acp_list: list[dict], field: str, idx: int = 0) -> str:
    if not acp_list or idx >= len(acp_list):
        return "—"
    return str(acp_list[idx].get(field, "—"))


def match_symbol(a: str, b: str) -> str:
    return f"{GREEN}✓{RESET}" if a == b else f"{RED}✗{RESET}"


def print_separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


def print_entry_comparison(
    idx: int,
    item: dict,
    predicted: list[dict],
    has_ground_truth: bool,
) -> dict:
    entry_id    = item.get("id", f"entry-{idx+1}")
    entry_class = item.get("class", "?")
    ground      = item.get("ACP", []) if has_ground_truth else []

    print_separator("═")
    print(f"{BOLD}[{idx+1}] ID: {entry_id}  |  Class: {entry_class}{RESET}")
    print_separator()

    # Conversation
    print(f"{CYAN}Conversation:{RESET}")
    for turn in item.get("NL", []):
        role = turn.get("role", "?")
        text = turn.get("text", "")
        print(f"  {BOLD}{role}:{RESET} {text}")
    print()

    n = max(len(predicted), len(ground) if ground else 0)
    matches = {"decision": None, "action": None}

    for i in range(n):
        if n > 1:
            print(f"  {YELLOW}── ACP entry {i+1} ──{RESET}")

        pred_entry   = predicted[i] if i < len(predicted) else {}
        ground_entry = ground[i]    if ground and i < len(ground) else {}

        # Flag unparsed entries clearly
        if "raw_output" in pred_entry:
            print(f"  {RED}[UNPARSED OUTPUT]{RESET}")
            print(f"  {pred_entry['raw_output'][:200]}")
            print()
            continue

        fields = ["acp_id", "decision", "action", "subject", "purpose",
                  "condition", "resource", "policy_metadata"]

        if has_ground_truth:
            col_w = 34
            print(f"  {'PREDICTED':<{col_w}}  {'GROUND TRUTH':<{col_w}}")
            print(f"  {'─'*col_w}  {'─'*col_w}")
            for field in fields:
                pval = str(pred_entry.get(field, "—"))
                gval = str(ground_entry.get(field, "—"))
                pval_disp = pval[:col_w-1] if len(pval) > col_w else pval
                gval_disp = gval[:col_w-1] if len(gval) > col_w else gval
                sym = match_symbol(pval, gval) if field in ("decision", "action") else " "
                print(f"  {field:<14} {pval_disp:<{col_w-15}}  {gval_disp:<{col_w-15}}  {sym}")

            if i == 0:
                matches["decision"] = (
                    pred_entry.get("decision") == ground_entry.get("decision")
                )
                matches["action"] = (
                    pred_entry.get("action") == ground_entry.get("action")
                )
        else:
            for field in fields:
                pval = str(pred_entry.get(field, "—"))
                print(f"  {field:<16} {pval}")

        print()

    # Quick-compare line
    if has_ground_truth and predicted and ground and "raw_output" not in predicted[0]:
        pred_dec = get_field(predicted, "decision")
        gt_dec   = get_field(ground,    "decision")
        pred_act = get_field(predicted, "action")
        gt_act   = get_field(ground,    "action")

        print(f"  {BOLD}Decision:{RESET} pred={pred_dec!r}  gt={gt_dec!r}  {match_symbol(pred_dec, gt_dec)}")
        print(f"  {BOLD}Action:  {RESET} pred={pred_act!r}  gt={gt_act!r}  {match_symbol(pred_act, gt_act)}")
        print()

    return matches


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------

def print_accuracy_report(
    match_log: list[dict],
    total: int,
    invalid_json: int,
) -> None:
    has_gt = [m for m in match_log if m["decision"] is not None]
    n = len(has_gt)

    print_separator("═")
    print(f"{BOLD}ACCURACY REPORT{RESET}")
    print_separator()
    print(f"  Total entries processed  : {total}")
    print(f"  Entries with ground truth: {n}")
    print(f"  Unparseable outputs      : {invalid_json}")
    print()

    if n == 0:
        print("  No ground-truth labels found — accuracy cannot be computed.")
        print_separator("═")
        return

    dec_correct  = sum(1 for m in has_gt if m["decision"] is True)
    act_correct  = sum(1 for m in has_gt if m["action"]   is True)
    both_correct = sum(1 for m in has_gt if m["decision"] is True and m["action"] is True)

    dec_acc  = dec_correct  / n * 100
    act_acc  = act_correct  / n * 100
    both_acc = both_correct / n * 100

    def bar(pct: float, width: int = 30) -> str:
        filled = int(pct / 100 * width)
        colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        return colour + "█" * filled + RESET + "░" * (width - filled)

    print(f"  Decision accuracy : {dec_correct:>4}/{n}  {bar(dec_acc)}  {dec_acc:6.1f}%")
    print(f"  Action   accuracy : {act_correct:>4}/{n}  {bar(act_acc)}  {act_acc:6.1f}%")
    print(f"  Both correct      : {both_correct:>4}/{n}  {bar(both_acc)}  {both_acc:6.1f}%")
    print()

    # Per-decision-value breakdown
    decision_buckets: dict[str, dict] = {}
    for m in has_gt:
        key = m.get("gt_decision", "unknown")
        decision_buckets.setdefault(key, {"correct": 0, "total": 0})
        decision_buckets[key]["total"] += 1
        if m["decision"]:
            decision_buckets[key]["correct"] += 1

    if decision_buckets:
        print(f"  {BOLD}Decision breakdown by ground-truth value:{RESET}")
        for val, counts in sorted(decision_buckets.items()):
            c, t = counts["correct"], counts["total"]
            pct  = c / t * 100
            print(f"    {val:<10} {c:>4}/{t}  {bar(pct, 20)}  {pct:6.1f}%")
        print()

    print_separator("═")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    logger.info("Reading: %s", args.input_json)
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = [item for item in data if item.get("NL")]
    logger.info("Loaded %d entries.\n", len(valid))

    has_ground_truth = any("ACP" in item for item in valid)

    predictor = ACPPredictor(
        model_dir=args.model_dir,
        device=args.device,
        max_input_len=args.max_input_len,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    all_predictions: list[list[dict]] = []
    for start in range(0, len(valid), args.batch_size):
        batch = valid[start : start + args.batch_size]
        preds = predictor.predict_batch([item["NL"] for item in batch])
        all_predictions.extend(preds)
        logger.info("Processed %d / %d", min(start + args.batch_size, len(valid)), len(valid))

    match_log: list[dict] = []
    invalid_json = 0
    results = []

    print("\n")
    for idx, (item, predicted) in enumerate(zip(valid, all_predictions)):
        if predicted and "raw_output" in predicted[0]:
            invalid_json += 1

        matches = print_entry_comparison(idx, item, predicted, has_ground_truth)

        if item.get("ACP"):
            matches["gt_decision"] = item["ACP"][0].get("decision", "unknown")
        match_log.append(matches)

        results.append({
            "id":            item.get("id"),
            "class":         item.get("class"),
            "NL":            item["NL"],
            "ACP_predicted": predicted,
            **({"ACP_ground_truth": item["ACP"]} if "ACP" in item else {}),
            "match":         matches,
        })

    print_accuracy_report(match_log, len(valid), invalid_json)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Predictions saved to: %s", args.output_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NL → ACP inference with side-by-side comparison and accuracy report"
    )
    parser.add_argument("--input_json",     type=str, default="/home/ubuntu/agentv-main/email_agent/dataset/val.json",
                        help="Input JSON file (train.json format)")
    parser.add_argument("--output_json",    type=str, default=None,
                        help="Optional: save predictions + matches to this file")
    parser.add_argument("--model_dir",      type=str, default="./nl_acp_model")
    parser.add_argument("--device",         type=str, default=None,
                        help="'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_input_len",  type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams",      type=int, default=4)

    args = parser.parse_args()
    main(args)
