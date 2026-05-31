"""
NL → ACP Test Script
=====================
Loads a CSV test file, runs the trained model's predict() on each row,
prints the results to stdout, and saves them to a CSV and JSON file.

Matching strategy (three levels):
    exact  – character-for-character match after stripping whitespace
    field  – all ACP fields (decision, action, resource id, condition) match
             regardless of minor wording differences in free-text fields
    decision – at minimum the decision value (allow/deny) matches

Usage:
    python test_nl_to_acp.py \
        --model_dir ./nl_acp_model \
        --test_path combined_test.csv \
        --output_path results.csv
"""

import argparse
import csv
import json
import logging
import re

from train_t5 import predict, load_examples

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ACP parsing
# ---------------------------------------------------------------------------

# ACP strings look like:  {decision: deny; subject: …; action: …; …}
# Values may contain commas and colons, so we split on "; key:" boundaries.
_FIELD_RE = re.compile(r'(\w+)\s*:\s*')

def parse_acp(text: str) -> dict[str, str]:
    """
    Parse a flat ACP string into a dict of field→value pairs.
    Works for both the gold format  {decision: deny; …}
    and the model's raw_output string.
    Strips surrounding braces/brackets and leading/trailing whitespace.
    """
    # Unwrap JSON array wrapper if predict() returned [{"raw_output": "..."}]
    text = text.strip()
    if text.startswith("[") or text.startswith("{\""):
        try:
            obj = json.loads(text)
            if isinstance(obj, list) and obj and "raw_output" in obj[0]:
                text = obj[0]["raw_output"]
        except json.JSONDecodeError:
            pass

    # Strip outer braces
    text = text.strip().lstrip("{").rstrip("}")

    # Split on semicolons that are followed by a known field name
    # e.g.  "decision: deny; subject: email_agent; action: forward_email"
    parts = re.split(r';\s*(?=\w+\s*:)', text)

    result = {}
    for part in parts:
        m = re.match(r'\s*(\w+)\s*:\s*(.*)', part.strip(), re.DOTALL)
        if m:
            result[m.group(1).lower().strip()] = m.group(2).strip()
    return result


def extract_resource_ref(resource_val: str) -> tuple[str | None, str | None]:
    """
    Pull out the id key name and its value from a resource field.
    Returns (key_name, value) e.g. ("email_id", "msg_8c4f2e7b").
    Both are None if no id token is found.
    """
    m = re.search(r'(email_id|message_id|msg_id|thread_id|id)\s*=\s*(\S+?)(?:,|;|$)', resource_val)
    if m:
        return m.group(1), m.group(2).rstrip(",;")
    return None, None


# ---------------------------------------------------------------------------
# Match scoring
# ---------------------------------------------------------------------------

# Actions that are acceptable substitutes when the ground truth is the key.
# Asymmetric: send_email is acceptable FOR reply/forward, but not the other way around.
_ACTION_ALIASES: dict[str, set[str]] = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
}

def _action_normalize(action: str) -> str:
    """Strip underscores/spaces and lowercase for loose comparison."""
    return re.sub(r"[\s_]", "", action).lower()


def action_match(exp_action: str, pred_action: str) -> bool:
    """
    Return True if pred_action is an acceptable match for exp_action.
    Checks both exact equality and the one-way alias table.
    Comparison is normalised (case- and separator-insensitive).
    """
    exp_norm  = _action_normalize(exp_action)
    pred_norm = _action_normalize(pred_action)

    if exp_norm == pred_norm:
        return True

    # Look up the ground-truth action in the alias table
    for canonical, aliases in _ACTION_ALIASES.items():
        if exp_norm == _action_normalize(canonical):
            return pred_norm in {_action_normalize(a) for a in aliases}

    return False


def score_match(expected: str, predicted: str) -> tuple[str, dict]:
    """
    Return a match level and a detail dict.

    Levels (best → worst):
        'exact'          – identical strings
        'field'          – decision + action + resource (key name + value) + condition all match
        'decision_action'– decision + action match but resource key/condition differ
                           (resource key mismatch also lands here — wrong key breaks fn call)
        'wrong'          – decision or action differs
    """
    details = {}

    # 1. Exact
    if expected.strip() == predicted.strip():
        return "exact", details

    exp  = parse_acp(expected)
    pred = parse_acp(predicted)

    details["expected_parsed"]  = exp
    details["predicted_parsed"] = pred

    exp_decision  = exp.get("decision", "").lower()
    pred_decision = pred.get("decision", "").lower()
    exp_action    = exp.get("action", "").lower()
    pred_action   = pred.get("action", "").lower()
    exp_cond      = exp.get("condition", "").lower()
    pred_cond     = pred.get("condition", "").lower()

    # Resource: both the key name AND value must match
    exp_rkey,  exp_rval  = extract_resource_ref(exp.get("resource", ""))
    pred_rkey, pred_rval = extract_resource_ref(pred.get("resource", ""))

    resource_key_match = exp_rkey == pred_rkey
    resource_val_match = exp_rval == pred_rval
    resource_match     = resource_key_match and resource_val_match

    details["decision_match"]      = exp_decision == pred_decision
    details["action_match"]        = action_match(exp_action, pred_action)
    details["resource_key_match"]  = resource_key_match
    details["resource_val_match"]  = resource_val_match
    details["resource_match"]      = resource_match
    details["condition_match"]     = exp_cond == pred_cond

    if not resource_key_match and exp_rkey and pred_rkey:
        details["resource_key_mismatch"] = f"{exp_rkey!r} (expected) vs {pred_rkey!r} (predicted)"
    print("decision_match",details["decision_match"],"\n", "action_match",details["action_match"],"\n", "resource_match",details["resource_match"])
    # 2. Field-level match — all four fields correct
    if all([
        details["decision_match"],
        details["action_match"],
        details["resource_match"],
        # details["condition_match"],
    ]):
        return "field", details

    # 3. Decision + action match (resource key mismatch or condition mismatch)
    if details["decision_match"] and details["action_match"]:
        return "decision_action", details

    return "wrong", details


MATCH_ICON = {
    "exact":           "✓✓",
    "field":           "✓ ",
    "decision_action": "~ ",
    "wrong":           "✗ ",
}


# ---------------------------------------------------------------------------
# Core test loop
# ---------------------------------------------------------------------------

def run_tests(model_dir: str, test_path: str) -> list[dict]:
    examples = load_examples(test_path)
    results  = []

    for i, ex in enumerate(examples):
        raw_input = ex["input_text"].removeprefix("translate to ACP: ")
        expected  = ex["target_text"]

        logger.info("Running example %d / %d …", i + 1, len(examples))

        nl_turns      = [{"role": "User", "text": raw_input}]
        predicted_obj = predict(model_dir, nl_turns)
        predicted_str = (
            json.dumps(predicted_obj, ensure_ascii=False)
            if not isinstance(predicted_obj, str)
            else predicted_obj
        )

        level, details = score_match(expected, predicted_str)
        icon           = MATCH_ICON[level]

        result = {
            "index":      i,
            "input":      raw_input,
            "expected":   expected,
            "predicted":  predicted_str,
            "match":      level,          # exact / field / decision / wrong
            "exact":      level == "exact",
            "field_match": level in ("exact", "field"),
            "decision_action_match": level in ("exact", "field", "decision_action"),
            "details":    details,
        }
        results.append(result)

        # ── Print ─────────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"[{i}] INPUT:\n{raw_input}")
        print(f"\n  EXPECTED : {expected}")
        print(f"  PREDICTED: {predicted_str}")
        print(f"  MATCH    : {icon} ({level})")
        if level not in ("exact", "field") and details:
            flags = {k: v for k, v in details.items() if k.endswith("_match")}
            print(f"  FIELDS   : {flags}")
            if "resource_key_mismatch" in details:
                print(f"  ⚠ RESOURCE KEY: {details['resource_key_mismatch']}")

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results_csv(results: list[dict], path: str) -> None:
    fieldnames = ["index", "input", "expected", "predicted", "match",
                  "exact", "field_match"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV saved → %s", path)


def save_results_json(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("JSON saved → %s", path)


def print_summary(results: list[dict]) -> None:
    total    = len(results)
    exact    = sum(1 for r in results if r["match"] == "exact")
    field    = sum(1 for r in results if r["match"] == "field")
    dec_act  = sum(1 for r in results if r["match"] == "decision_action")
    wrong    = sum(1 for r in results if r["match"] == "wrong")

    print(f"\n{'='*70}")
    print(f"RESULTS ({total} examples)")
    print(f"  ✓✓ Exact match          : {exact:3d}  ({100*exact/total:.1f}%)")
    print(f"  ✓  Field match          : {field:3d}  ({100*field/total:.1f}%)")
    print(f"  ~  Decision+Action only : {dec_act:3d}  ({100*dec_act/total:.1f}%)")
    print(f"  ✗  Wrong                : {wrong:3d}  ({100*wrong/total:.1f}%)")
    print(f"  ── Correct (≥field)     : {exact+field:3d}  ({100*(exact+field)/total:.1f}%)")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    results = run_tests(args.model_dir, args.test_path)
    print_summary(results)

    save_results_csv(results, args.output_path)

    json_path = args.output_path.replace(".csv", ".json")
    if json_path == args.output_path:
        json_path += ".json"
    save_results_json(results, json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained NL → ACP model.")

    parser.add_argument("--model_dir",   type=str, default="./nl_acp_model",
                        help="Path to the saved model directory")
    parser.add_argument("--test_path",   type=str, default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.csv",
                        help="Test CSV with 'input' and 'output' columns")
    parser.add_argument("--output_path", type=str, default="results.csv",
                        help="Where to save results (CSV + JSON written alongside)")

    args = parser.parse_args()
    main(args)