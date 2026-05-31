import ast
import re
import numpy as np
from evaluate import load
from transformers import EvalPrediction

# ===========================================================================
# Metrics (unchanged)
# ===========================================================================

def compute_metrics(eval_pred: EvalPrediction):
    load_accuracy = load("accuracy")
    load_f1 = load("f1")

    logits = eval_pred.predictions[0]
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    accuracy     = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_micro     = load_f1.compute(predictions=predictions, references=labels, average="micro")["f1"]
    f1_macro     = load_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    f1_weighted  = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {
        "accuracy":    accuracy,
        "f1-micro":    f1_micro,
        "f1-macro":    f1_macro,
        "f1-weighted": f1_weighted,
    }


# ===========================================================================
# ACP serialisation  (used by component_manipulation.py)
# ===========================================================================

def create_out_string(inp: list[dict]) -> str:
    """Serialise a list of ACP rule dicts to the canonical {k: v; …} | … format."""
    s = "{"
    for e in inp:
        for k, v in e.items():
            s += f"{k}: {v}; "
        s = s[:-2] + " | "
    return s[:-3] + "}"


# ===========================================================================
# ACP parsing  (single source of truth, shared with test_t5 / model_generation)
# ===========================================================================

def parse_acp(text: str) -> dict:
    """
    Parse a flat ACP string into a dict of field→value pairs.

    Handles:
      • {decision: deny; subject: …; action: …; …}
      • JSON raw_output wrapper from predict()
    """
    import json

    text = text.strip()
    # Unwrap JSON array wrapper if predict() returned [{"raw_output": "..."}]
    if text.startswith("[") or text.startswith('{"'):
        try:
            obj = json.loads(text)
            if isinstance(obj, list) and obj and "raw_output" in obj[0]:
                text = obj[0]["raw_output"]
        except json.JSONDecodeError:
            pass

    text = text.strip().lstrip("{").rstrip("}")
    parts = re.split(r";\s*(?=\w+\s*:)", text)
    result = {}
    for part in parts:
        m = re.match(r"\s*(\w+)\s*:\s*(.*)", part.strip(), re.DOTALL)
        if m:
            result[m.group(1).lower().strip()] = m.group(2).strip()
    return result


def extract_resource_ref(resource_val: str) -> tuple:
    """
    Pull out the id key name and value from a resource field.
    Returns (key_name, value), both None if no id token is found.
    """
    m = re.search(
        r"(email_id|message_id|msg_id|thread_id|id)\s*=\s*(\S+?)(?:,|;|$)",
        resource_val,
    )
    if m:
        return m.group(1), m.group(2).rstrip(",;")
    return None, None


_ACTION_ALIASES: dict = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
    "send_email":    {"send_email",    "forward_email"},
}


def _action_normalize(action: str) -> str:
    return re.sub(r"[\s_]", "", action).lower()


def action_match(exp_action: str, pred_action: str) -> bool:
    """Return True if pred_action is an acceptable substitute for exp_action."""
    exp_norm  = _action_normalize(exp_action)
    pred_norm = _action_normalize(pred_action)
    if exp_norm == pred_norm:
        return True
    for canonical, aliases in _ACTION_ALIASES.items():
        if exp_norm == _action_normalize(canonical):
            return pred_norm in {_action_normalize(a) for a in aliases}
    return False


def is_valid_segment(segment: str) -> bool:
    """Return True if this ACP segment has both a decision and an action."""
    parsed = parse_acp(segment.strip())
    return bool(parsed.get("decision", "").strip() and parsed.get("action", "").strip())


def clean_gold_segments(output: str) -> list:
    """
    Split a gold output on '|' and return only the valid ACP segments.
    Drops garbage fragments like 'decision: allow.' or trailing notes.
    """
    return [s.strip() for s in output.split("|") if s.strip() and is_valid_segment(s)]


def is_valid_gold(output: str) -> bool:
    """
    Return True only if the gold output contains at least one parseable ACP
    with a non-empty decision and action field.
    """
    if not output or not output.strip():
        return False
    return len(clean_gold_segments(output)) > 0


def is_field_match(expected: str, predicted: str) -> bool:
    """
    Return True when decision + action + resource (key name + value) all match.
    For multi-ACP gold outputs (pipe-separated), a match against ANY segment counts.
    """
    if not expected.strip() or not predicted.strip():
        return False
    if expected.strip() == predicted.strip():
        return True

    gold_segments = clean_gold_segments(expected)
    if not gold_segments:
        return False

    for gold in gold_segments:
        exp  = parse_acp(gold)
        pred = parse_acp(predicted)

        exp_decision  = exp.get("decision", "").lower()
        pred_decision = pred.get("decision", "").lower()
        exp_action    = exp.get("action",   "").lower()
        pred_action   = pred.get("action",  "").lower()

        exp_rkey,  exp_rval  = extract_resource_ref(exp.get("resource",  ""))
        pred_rkey, pred_rval = extract_resource_ref(pred.get("resource", ""))

        if (
            exp_decision == pred_decision
            and action_match(exp_action, pred_action)
            and exp_rkey == pred_rkey
            and exp_rval == pred_rval
        ):
            return True

    return False


def score_match(expected: str, predicted: str) -> tuple:
    """
    Return a (level, details) tuple.

    Levels (best → worst):
        'exact'          – identical strings
        'field'          – decision + action + resource (key + value) all match
        'decision_action'– decision + action match, resource/condition differ
        'wrong'          – decision or action differs
    """
    details = {}

    if expected.strip() == predicted.strip():
        return "exact", details

    exp  = parse_acp(expected)
    pred = parse_acp(predicted)

    details["expected_parsed"]  = exp
    details["predicted_parsed"] = pred

    exp_decision  = exp.get("decision",  "").lower()
    pred_decision = pred.get("decision", "").lower()
    exp_action    = exp.get("action",    "").lower()
    pred_action   = pred.get("action",   "").lower()
    exp_cond      = exp.get("condition", "").lower()
    pred_cond     = pred.get("condition","").lower()

    exp_rkey,  exp_rval  = extract_resource_ref(exp.get("resource",  ""))
    pred_rkey, pred_rval = extract_resource_ref(pred.get("resource", ""))

    resource_key_match = exp_rkey == pred_rkey
    resource_val_match = exp_rval == pred_rval
    resource_match     = resource_key_match and resource_val_match

    details["decision_match"]     = exp_decision == pred_decision
    details["action_match"]       = action_match(exp_action, pred_action)
    details["resource_key_match"] = resource_key_match
    details["resource_val_match"] = resource_val_match
    details["resource_match"]     = resource_match
    details["condition_match"]    = exp_cond == pred_cond

    if not resource_key_match and exp_rkey and pred_rkey:
        details["resource_key_mismatch"] = (
            f"{exp_rkey!r} (expected) vs {pred_rkey!r} (predicted)"
        )

    if all([details["decision_match"], details["action_match"], details["resource_match"]]):
        return "field", details

    if details["decision_match"] and details["action_match"]:
        return "decision_action", details

    return "wrong", details


# ===========================================================================
# Legacy ACP parsing helpers  (used internally by process_label / make_json)
# These are kept for component_manipulation.py which relies on them, but the
# canonical parse_acp() above should be preferred for new code.
# ===========================================================================

a, count = 0, 0
failed_parse = []


def format_labels(label: str) -> str:
    nlabel = (
        label.lower()
        .replace('"', "")
        .replace(";", ",")
        .replace("decision: ",  "'decision': '")
        .replace("subject: ",   "'subject': '")
        .replace("action: ",    "'action': '")
        .replace("resource: ",  "'resource': '")
        .replace("condition: ", "'condition': '")
        .replace("purpose: ",   "'purpose': '")
        .replace(",",           "',")
        .replace("}",           "'}")
        .replace("'s ",         "\\'s ")
    )
    i = nlabel.find("decision")
    return "{'" + nlabel[i:]


def make_json(labels: list) -> list:
    global count, a, failed_parse
    policies = []
    formatted_list = []
    for l in labels:
        formatted = format_labels(l)
        formatted_list.append(formatted)

    for f in list(set(formatted_list)):
        a += 1
        try:
            label_json = ast.literal_eval(f)
            pp = {
                "decision": "allow",
                "subject":  "none",
                "action":   "none",
                "resource": "none",
                "condition":"none",
                "purpose":  "none",
            }
            for key, val in label_json.items():
                if key in pp:
                    pp[key] = val.strip()
            policies.append(pp)
        except Exception:
            count += 1
            failed_parse.append([labels, f])
            continue

    p = []
    for pol in policies:
        if pol not in p:
            p.append(pol)
    return p


def process_label(result: list) -> list:
    res = []
    if len(result) > 0:
        for p in result:
            ind = p.split(" | ")
            if len(ind) == 1:
                res.append(ind[0])
            else:
                for i in range(len(ind)):
                    if i == 0 and ind[i][-1] != "}":
                        res.append(ind[i] + "}")
                    elif i == len(ind) - 1 and ind[i][0] != "{":
                        res.append("{" + ind[i])
                    else:
                        res.append("{" + ind[i] + "}")
    nres = list(set(res))
    return make_json(nres)


# ===========================================================================
# String similarity helpers  (unchanged)
# ===========================================================================

def longest_common_substring(str1: str, str2: str):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    longest_length = 0
    longest_end    = 0

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_length:
                    longest_length = dp[i][j]
                    longest_end    = i - 1
            else:
                dp[i][j] = 0

    start = longest_end - longest_length + 1
    return str1[start : longest_end + 1], start, longest_end


def do_overlap(x: str, y: str, thresh: float = 1.0) -> bool:
    overlap, start, end = longest_common_substring(x.lower(), y.lower())
    seg_len = end - start + 1
    return (seg_len / len(x)) >= thresh


def is_equal(preds: list, labels: list) -> bool:
    pcopy = preds.copy()
    lcopy = labels.copy()
    found = []

    if len(preds) != len(labels):
        return False

    for pred in preds:
        if pred in labels and pred not in found:
            found.append(pred)
            lcopy.remove(pred)
            pcopy.remove(pred)

    for pred in preds:
        if pred not in found:
            d, s, a, r, p, c = (
                pred["decision"], pred["subject"], pred["action"],
                pred["resource"], pred["purpose"], pred["condition"],
            )
            for l in labels:
                if l in found:
                    continue
                dl, sl, al, rl, pl, cl = (
                    l["decision"], l["subject"], l["action"],
                    l["resource"], l["purpose"], l["condition"],
                )
                if (
                    do_overlap(dl, d) and do_overlap(sl, s) and do_overlap(al, a)
                    and do_overlap(rl, r, 0.8) and do_overlap(pl, p, 0.2)
                    and do_overlap(cl, c, 0.2)
                ):
                    found.append(l)
                    lcopy.remove(l)
                    pcopy.remove(pred)
                    break

    return len(pcopy) == len(lcopy) == 0


def prepare_inputs_bart(s, l, tokenizer, device="cuda:0"):
    tokens = tokenizer(s, l, return_tensors="pt")
    return {k: v.to(device) for k, v in tokens.items()}