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

from train_t5 import predict, load_examples, predict_base_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


REQUIRED_ARGS: dict[str, list[str]] = {
    "list_emails": [],
    "get_email": ["email_id"],
    "search_emails": ["query"],
    "analyze_email": ["email_id"],
    "draft_reply": ["email_id"],
    "send_email": ["to", "subject", "body"],
    "reply_email": ["to", "body"],
    "send_draft": ["draft_id"],
    "draft_email": ["to", "subject", "body"],
    "forward_email": ["email_id", "to"],
    "delete_email": ["email_id"],
    "star_email": ["email_id"],
    "summarize_inbox": [],
    "create_task": ["title"],
    "list_tasks": [],
    "complete_task": ["task_id"],
}

# Free-text fields scored by SEMANTIC similarity (>= threshold = match).
# Everything else is scored by exact (normalized) equality.
SEMANTIC_FIELDS = {
    "query", "tone", "instructions", "subject",
    "body", "focus", "description", "message",
}

SIM_THRESHOLD = 0.75

# Action aliases: actions acceptable as substitutes for a ground-truth action.
# Keyed by the GROUND-TRUTH action; every action implicitly matches itself.
# Asymmetric by design (send_email can stand in for reply/forward).
_ACTION_ALIASES: dict[str, set[str]] = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
    "send_email":    {"send_email",    "forward_email"},
}

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
    m = re.search(r'(email_id|message_id|msg_id|thread_id|id|subject|to|body|reply_to_id|draft_id|focus|max_results|description|title|deadline|email_id|email_subject|priority|status|task_id|message|star|query)\s*=\s*(\S+?)(?:,|;|$)', resource_val)
    if m:
        return m.group(1), m.group(2).rstrip(",;")
    return None, None


# ---------------------------------------------------------------------------
# Match scoring
# ---------------------------------------------------------------------------

_EMBED_CACHE: dict[str, list[float]] = {}


def _difflib_sim(a: str, b: str) -> float:
    """Offline fallback similarity (used when embeddings are unavailable)."""
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _cosine(u: list[float], v: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def semantic_similarity(client, embed_model: str, a: str, b: str) -> float:
    """Cosine similarity of OpenAI embeddings; difflib fallback if unavailable.
    Returns a value in [0, 1]."""
    a, b = (a or "").strip(), (b or "").strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if client is None:
        return _difflib_sim(a.lower(), b.lower())

    try:
        to_fetch = [t for t in (a, b) if t not in _EMBED_CACHE]
        if to_fetch:
            resp = client.embeddings.create(model=embed_model, input=to_fetch)
            for text, item in zip(to_fetch, resp.data):
                _EMBED_CACHE[text] = item.embedding
        return max(0.0, min(1.0, _cosine(_EMBED_CACHE[a], _EMBED_CACHE[b])))
    except Exception:  # noqa: BLE001
        return _difflib_sim(a.lower(), b.lower())

# Actions that are acceptable substitutes when the ground truth is the key.
# Asymmetric: send_email is acceptable FOR reply/forward, but not the other way around.
_ACTION_ALIASES: dict[str, set[str]] = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
    "send_email": {"send_email", "forward_email"},
}

def _action_normalize(action: str) -> str:
    """Strip underscores/spaces and lowercase for loose comparison."""
    return re.sub(r"[\s_]", "", action).lower()


def action_matches(exp_action: str, pred_action: str) -> bool:
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


def norm_value(val) -> object:
    """Normalize a scalar value for exact comparison.
    Strings: lower-cased + whitespace-collapsed. Numbers: compared by value."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"\s+", " ", str(val).strip().lower())
    # Treat a numeric-looking string as a number so "10" == 10.
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return s

def parse_resource_args(resource_val: str) -> dict[str, str]:
    """
    Parse an ACP resource string into an {arg_key: value} dict, mirroring the
    CODE "args" object.

    ACP resource looks like:  "query: is:unread, max_results: 10"
    or sometimes:             "email_id=msg_8c4f2e7b, to=alice@x.com"

    Splits on commas that precede a "key: " or "key=" token so that values
    containing colons/commas (e.g. a body) are not split mid-value.
    """
    resource_val = (resource_val or "").strip()
    if not resource_val:
        return {}

    # Split on commas that introduce a new "key:" or "key=" pair.
    parts = re.split(r',\s*(?=\w+\s*[:=])', resource_val)
    args: dict[str, str] = {}
    for part in parts:
        m = re.match(r'\s*(\w+)\s*[:=]\s*(.*)', part.strip(), re.DOTALL)
        if m:
            args[m.group(1).lower().strip()] = m.group(2).strip().rstrip(",;")
    return args
def compare_resource(client, embed_model: str, gen_action: str, gt_action: str,
                     gen_args: dict, gt_args: dict) -> dict:
    """Compare the REQUIRED args of the GENERATED action.

    Schema is driven by the GENERATED action (important when an alias was used,
    e.g. gen=send_email standing in for gt=forward_email). We validate every
    required key of gen_action for which ground truth provides a value (the
    intersection of gen_action's required args and the keys present in gt_args).
    Keys required by gen_action but absent from ground truth are skipped.

    Free-text fields use semantic similarity (>= SIM_THRESHOLD = match);
    all other required fields use exact normalized equality.
    Optional args are ignored entirely.
    """
    gen_args = gen_args or {}
    gt_args = gt_args or {}

    if gen_action in REQUIRED_ARGS:
        required = REQUIRED_ARGS[gen_action]
    elif gt_action in REQUIRED_ARGS:
        required = REQUIRED_ARGS[gt_action]
    else:
        required = list(gt_args.keys())

    field_results = {}
    all_ok = True
    checked_any = False
    for key in required:
        if key not in gt_args:
            field_results[key] = {"type": "skipped", "match": None,
                                  "note": "no ground-truth value"}
            continue
        checked_any = True
        gen_val = gen_args.get(key)
        gt_val = gt_args.get(key)

        if key in SEMANTIC_FIELDS:
            sim = semantic_similarity(
                client, embed_model, str(gen_val or ""), str(gt_val or ""))
            ok = sim >= SIM_THRESHOLD
            field_results[key] = {"type": "semantic", "sim": round(sim, 3),
                                  "match": ok}
        else:
            ok = norm_value(gen_val) == norm_value(gt_val)
            field_results[key] = {"type": "exact", "match": ok}
        all_ok = all_ok and ok

    return {"match": all_ok, "fields": field_results, "checked": checked_any}

def _action_normalize_to_canon(action: str) -> str:
    """
    Map a (possibly spaced/cased) action name to a key usable in REQUIRED_ARGS.
    REQUIRED_ARGS uses snake_case lower keys; the model may emit spaces, so we
    collapse spaces to underscores and lowercase. Falls back to the raw lower
    form if no match is found.
    """
    raw = (action or "").strip().lower()
    snake = re.sub(r"\s+", "_", raw)
    if snake in REQUIRED_ARGS:
        return snake
    if raw in REQUIRED_ARGS:
        return raw
    return snake

def score_match(client,embed_model,expected: str, predicted: str) -> tuple[str, dict]:
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

    exp_decision = exp.get("decision", "").strip().lower()
    pred_decision = pred.get("decision", "").strip().lower()
    exp_action = exp.get("action", "").strip()
    pred_action = pred.get("action", "").strip()

    exp_args = parse_resource_args(exp.get("resource", ""))
    pred_args = parse_resource_args(pred.get("resource", ""))

    res = compare_resource(client, embed_model,
                           _action_normalize_to_canon(pred_action),
                           _action_normalize_to_canon(exp_action),
                           pred_args, exp_args)
    print("$$",exp_args)
    print("$$",pred_args)

    details["decision_match"] = exp_decision == pred_decision
    details["action_match"] = action_matches(pred_action, exp_action)
    details["action_alias"] = (details["action_match"]
                               and _action_normalize(pred_action)
                               != _action_normalize(exp_action))
    details["resource_match"] = res["match"]
    details["resource_fields"] = res["fields"]
    details["resource_checked"] = res["checked"]

    print("decision_match", details["decision_match"], "\n",
          "action_match", details["action_match"], "\n",
          "resource_match", details["resource_match"])

    # 2. Field-level match — decision + action + resource(required args) correct
    if all([
        details["decision_match"],
        details["action_match"],
        details["resource_match"],
    ]):
        return "field", details

    # 3. Decision + action match (resource differs)
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

def run_tests(model_dir: str, test_path: str, client, embed_model: str) -> list[dict]:
    examples = load_examples(test_path)
    results  = []

    for i, ex in enumerate(examples):
        raw_input = ex["input_text"].removeprefix("translate to ACP: ")
        expected  = ex["target_text"]

        logger.info("Running example %d / %d …", i + 1, len(examples))

        nl_turns      = [{"role": "User", "text": raw_input}]
        predicted_obj = predict_base_model(model_dir, nl_turns)
        predicted_str = (
            json.dumps(predicted_obj, ensure_ascii=False)
            if not isinstance(predicted_obj, str)
            else predicted_obj
        )

        level, details = score_match(client, embed_model,expected, predicted_str)
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
    total = len(results) or 1
    exact = sum(1 for r in results if r["match"] == "exact")
    field = sum(1 for r in results if r["match"] == "field")
    dec_act = sum(1 for r in results if r["match"] == "decision_action")
    wrong = sum(1 for r in results if r["match"] == "wrong")
    decision_match = sum(1 for r in results
                         if r["details"].get("decision_match"))
    api_match      = sum(1 for r in results
                         if r["details"].get("action_match"))
    args_match     = sum(1 for r in results
                         if r["details"].get("resource_match"))

    print(f"\n{'='*70}")
    print(f"RESULTS ({len(results)} examples)")
    print(f"  Exact match          : {exact:3d}  ({100*exact/total:.1f}%)")
    print(f"  Field match          : {field:3d}  ({100*field/total:.1f}%)")
    print(f"  Decision+API only    : {dec_act:3d}  ({100*dec_act/total:.1f}%)")
    print(f"  Wrong                : {wrong:3d}  ({100*wrong/total:.1f}%)")
    print(f"  decision_match       : {decision_match:3d}  ({100*decision_match/total:.1f}%)")
    print(f"  api_match            : {api_match:3d}  ({100*api_match/total:.1f}%)")
    print(f"  args_match           : {args_match:3d}  ({100*args_match/total:.1f}%)")
    print(f"  Correct (>=field)    : {exact+field:3d}  ({100*(exact+field)/total:.1f}%)")
    print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"RESULTS ({len(results)} examples)")
    print(f"  Exact match          : {exact:3d}  ({100*exact/total:.1f}%)")
    print(f"  Field match          : {field:3d}  ({100*field/total:.1f}%)")
    print(f"  Decision+API only    : {dec_act:3d}  ({100*dec_act/total:.1f}%)")
    print(f"  Wrong                : {wrong:3d}  ({100*wrong/total:.1f}%)")
    print(f"  decision_match       : {decision_match:3d}  ({100*decision_match/total:.1f}%)")
    print(f"  api_match            : {api_match:3d}  ({100*api_match/total:.1f}%)")
    print(f"  args_match           : {args_match:3d}  ({100*args_match/total:.1f}%)")
    print(f"  Correct (>=field)    : {exact+field:3d}  ({100*(exact+field)/total:.1f}%)")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    results = run_tests(args.model_dir, args.test_path,None,args.embed_model)
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
    parser.add_argument("--test_path",   type=str, default="/home/ubuntu/agentv-main/email_agent/dataset/combined_no0_test.csv",
                        help="Test CSV with 'input' and 'output' columns")
    parser.add_argument("--output_path", type=str, default="results.csv",
                        help="Where to save results (CSV + JSON written alongside)")
    parser.add_argument("--embed-model", dest="embed_model", type=str,
                        default="text-embedding-3-small",
                        help="OpenAI embedding model for semantic field matching")

    args = parser.parse_args()
    main(args)