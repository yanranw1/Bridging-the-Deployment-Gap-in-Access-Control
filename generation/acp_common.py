"""
acp_common.py
=============
Single source of truth for everything the NL→ACP train/test scripts share:

  * the tool catalogue (REQUIRED_ARGS, SEMANTIC_FIELDS, action aliases)
  * parsing a flat ACP string  ->  a structured dict
  * serializing a structured dict  ->  a clean JSON target the model learns
  * structured, field-level scoring (decision / action / required-args)

WHY THIS FILE EXISTS
--------------------
Originally train_t5 emitted the *flat* ACP string as the target and test_t5
re-parsed that string with brittle regexes at scoring time. That made the model
generate a delimiter-heavy string (hard for the T5 tokenizer) AND scored it by
re-parsing, so punctuation drift turned correct predictions into "wrong". By
giving the model a JSON target and scoring structurally (exactly like the
NL→CODE pipeline in test.py), the ACP pipeline is judged on the same footing.

The on-disk CSV format is unchanged: the `output` column is still the flat ACP
string. We parse it to structured form once, at load time.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Catalogue (kept identical to test.py / eval_common.py)
# ---------------------------------------------------------------------------

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
SEMANTIC_FIELDS = {
    "query", "tone", "instructions", "subject",
    "body", "focus", "description", "message", "title",
}

# Action aliases keyed by GROUND-TRUTH action. Asymmetric by design.
_ACTION_ALIASES: dict[str, set[str]] = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
    "send_email":    {"send_email",    "forward_email"},
}

# Top-level ACP keys we model. `subject` is constant ("email_agent") and
# `purpose`/`condition` are free-form rationale we do NOT score, but we keep
# them in the target so the model still learns to produce a complete object.
ACP_TOP_KEYS = ["decision", "action", "resource", "subject", "purpose", "condition"]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def action_normalize(action: str) -> str:
    """Collapse spaces/underscores and lowercase for loose action comparison."""
    return re.sub(r"[\s_]", "", action or "").lower()


def canon_action(action: str) -> str:
    """Map a (spaced/cased) action to a REQUIRED_ARGS key (snake_case)."""
    raw = (action or "").strip().lower()
    snake = re.sub(r"\s+", "_", raw)
    if snake in REQUIRED_ARGS:
        return snake
    if raw in REQUIRED_ARGS:
        return raw
    return snake


def action_matches(gen_action: str, gt_action: str) -> bool:
    """Action equals ground truth, or is an accepted alias of it."""
    g = action_normalize(gen_action)
    t = action_normalize(gt_action)
    if g == t:
        return True
    for canonical, aliases in _ACTION_ALIASES.items():
        if t == action_normalize(canonical):
            return g in {action_normalize(a) for a in aliases}
    return False


def norm_value(val: Any) -> Any:
    """Normalise a scalar for exact comparison (numbers by value, strings
    lower+whitespace-collapsed, numeric-looking strings as numbers)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"\s+", " ", str(val).strip().lower())
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return s


# ---------------------------------------------------------------------------
# Flat ACP string  <->  structured dict
# ---------------------------------------------------------------------------

def _split_top_level(body: str) -> dict[str, str]:
    """Split 'decision: x; action: y; resource: ...' on ';' before a known key."""
    parts = re.split(r";\s*(?=\w+\s*:)", body)
    out: dict[str, str] = {}
    for part in parts:
        m = re.match(r"\s*(\w+)\s*:\s*(.*)", part.strip(), re.DOTALL)
        if m:
            out[m.group(1).lower().strip()] = m.group(2).strip()
    return out


def parse_resource_args(resource_val: str) -> dict[str, str]:
    """Parse 'query: is:unread, max_results: 10' -> {'query': ..., ...}.
    Splits on commas that introduce a new 'key:' or 'key=' pair so values
    containing commas/colons (e.g. a body) are not split mid-value."""
    resource_val = (resource_val or "").strip()
    if not resource_val:
        return {}
    parts = re.split(r",\s*(?=\w+\s*[:=])", resource_val)
    args: dict[str, str] = {}
    for part in parts:
        m = re.match(r"\s*(\w+)\s*[:=]\s*(.*)", part.strip(), re.DOTALL)
        if m:
            args[m.group(1).lower().strip()] = m.group(2).strip().rstrip(",;")
    return args


def parse_flat_acp(text: str) -> dict[str, Any]:
    """Flat ACP string -> structured dict with a real `resource` sub-dict.

    Returns {} for an empty/`{}` ACP (the deny-and-do-nothing case)."""
    text = (text or "").strip()
    if not text or text == "{}" or text.upper() == "NONE":
        return {}
    body = text.lstrip("{").rstrip("}")
    top = _split_top_level(body)
    if not top:
        return {}
    resource = parse_resource_args(top.get("resource", ""))
    obj: dict[str, Any] = {
        "decision": top.get("decision", "").strip(),
        "action": top.get("action", "").strip(),
        "resource": resource,
    }
    # Keep optional rationale fields if present (not scored, but learned).
    for k in ("subject", "purpose", "condition"):
        if k in top:
            obj[k] = top[k].strip()
    return obj


def serialize_acp_target(obj: dict[str, Any]) -> str:
    """Structured ACP dict -> the string the model learns to emit.

    IMPORTANT: T5's SentencePiece vocabulary has no '{' or '}' tokens, so a JSON
    target is impossible for the model to reproduce — it drops every brace and
    the output becomes unparseable. We therefore emit a BRACE-FREE flat format
    using only in-vocab delimiters (';', ':', ',', '='):

        decision: allow; action: forward_email; resource: email_id=msg_1, to=a@b.com; subject: email_agent; purpose: ...; condition: ...

    This is the same shape as the gold CSV `output`, so parse_flat_acp round-
    trips it directly. Empty object (deny-and-nothing) -> 'NONE' (also brace-free
    and a single in-vocab token), which parse_flat_acp maps back to {}."""
    if not obj:
        return "NONE"
    parts: list[str] = []
    parts.append(f"decision: {obj.get('decision', '')}")
    parts.append(f"action: {obj.get('action', '')}")
    resource = obj.get("resource", {}) or {}
    res_str = ", ".join(f"{k}={v}" for k, v in resource.items())
    parts.append(f"resource: {res_str}")
    for k in ("subject", "purpose", "condition"):
        if k in obj:
            parts.append(f"{k}: {obj[k]}")
    return "; ".join(parts)


def parse_prediction(text: str) -> dict[str, Any]:
    """Model output -> structured ACP dict.

    Tries (1) direct JSON, (2) outermost {...} JSON, (3) flat-string fallback
    via parse_flat_acp. The T5 tokenizer can still drop braces, so the
    flat-string fallback recovers those cases instead of scoring them wrong."""
    text = (text or "").strip()
    if not text or text == "{}" or text.upper() == "NONE":
        return {}

    # 1. Direct JSON (for older JSON-trained checkpoints). Try as-is, then a
    #    brace-balanced wrap (recovers the case where only the OUTER braces are
    #    dropped), then the outermost {...} span as a last resort.
    candidates = [text]
    bal = text.count("{") - text.count("}")
    wrapped = text
    if not text.startswith("{"):
        wrapped = "{" + wrapped
        bal += 1
    if bal > 0:
        wrapped = wrapped + "}" * bal
    if wrapped != text:
        candidates.append(wrapped)
    candidates.append(_outermost_braces(text))

    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                norm = _normalize_pred_dict(obj)
                # An all-empty parse is treated as the deny-and-nothing case.
                if not norm["decision"] and not norm["action"] and not norm["resource"]:
                    return {}
                return norm
        except json.JSONDecodeError:
            pass

    # 2. Braceless-stream recovery. T5's SentencePiece vocab has no '{' or '}',
    #    so the model often emits a quoted key:value stream with ALL braces
    #    dropped, including the resource sub-object's braces:
    #        "decision":"allow","action":"...","resource":"email_id":"...",...
    #    This is unparseable as JSON, so reconstruct the structure from the
    #    key stream, routing resource args into a sub-dict.
    braceless = _recover_braceless(text)
    if braceless:
        return braceless

    # 3. Flat-string fallback for the gold-style "decision: x; action: y; ..."
    return parse_flat_acp(text)


# Top-level ACP keys vs. keys that belong inside the resource sub-object.
_TOP_LEVEL_KEYS = {"decision", "action", "resource", "subject", "purpose", "condition"}


def _recover_braceless(text: str) -> dict[str, Any]:
    """Reconstruct an ACP dict from a brace-stripped quoted key:value stream.

    Splits on every `"key":` boundary so values may contain commas, colons, '@'
    etc. The `resource` token opens a sub-object that runs until the next
    top-level key (purpose/condition/subject after the resource block, or a
    second decision/action). `subject` is ambiguous (valid both top-level and in
    a few actions) — once the resource block has opened we only close it on
    purpose/condition, the keys that always follow resource in this schema."""
    segments = re.split(r'"(\w+)"\s*:', text)
    # segments[0] is leading junk; then [key, chunk, key, chunk, ...]
    pairs: list[tuple[str, str]] = []
    for i in range(1, len(segments), 2):
        key = segments[i]
        raw = segments[i + 1] if i + 1 < len(segments) else ""
        val = raw.strip().strip(",").strip()
        if val.startswith('"'):
            val = val[1:]
        if val.endswith('"'):
            val = val[:-1]
        pairs.append((key, val))

    if not pairs:
        return {}

    obj: dict[str, Any] = {}
    resource: dict[str, str] = {}
    in_resource = False
    # Keys that definitively close the resource block (always trail it here).
    _closers = {"purpose", "condition"}
    for key, val in pairs:
        if key == "resource":
            in_resource = True
            continue
        if in_resource and key not in _closers:
            resource[key] = val
        else:
            in_resource = False
            obj[key] = val

    if resource:
        obj["resource"] = resource
    if not obj.get("decision") and not obj.get("action") and not resource:
        return {}
    return _normalize_pred_dict(obj)


def _outermost_braces(text: str) -> str:
    bs, be = text.find("{"), text.rfind("}")
    if bs != -1 and be > bs:
        return text[bs:be + 1]
    return ""


def _normalize_pred_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parsed dict to the {decision, action, resource{...}} shape.
    Accepts `resource` as a dict or a flat string."""
    res = obj.get("resource", {})
    if isinstance(res, str):
        res = parse_resource_args(res)
    elif not isinstance(res, dict):
        res = {}
    return {
        "decision": str(obj.get("decision", "")).strip(),
        "action": str(obj.get("action", "")).strip(),
        "resource": res,
    }


# ---------------------------------------------------------------------------
# Semantic similarity (OpenAI embeddings with offline difflib fallback)
# ---------------------------------------------------------------------------

_EMBED_CACHE: dict[str, list[float]] = {}


def _difflib_sim(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _cosine(u: list[float], v: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def semantic_similarity(client, embed_model: str, a: str, b: str) -> float:
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


# ---------------------------------------------------------------------------
# Structured scoring (mirrors test.py's resource_matches / per-step logic)
# ---------------------------------------------------------------------------

def compare_resource(client, embed_model: str, sim_threshold: float,
                     gen_action: str, gt_action: str,
                     gen_args: dict, gt_args: dict) -> dict:
    """Compare REQUIRED args of the GENERATED action (alias-aware).
    Required keys with no ground-truth value are skipped. Free-text fields use
    semantic similarity; all others use exact normalised equality."""
    gen_args = gen_args or {}
    gt_args = gt_args or {}

    g = canon_action(gen_action)
    t = canon_action(gt_action)
    if g in REQUIRED_ARGS:
        required = REQUIRED_ARGS[g]
    elif t in REQUIRED_ARGS:
        required = REQUIRED_ARGS[t]
    else:
        required = list(gt_args.keys())

    fields: dict[str, Any] = {}
    all_ok = True
    checked = False
    for key in required:
        if key not in gt_args:
            fields[key] = {"type": "skipped", "match": None}
            continue
        checked = True
        gen_val, gt_val = gen_args.get(key), gt_args.get(key)
        if key in SEMANTIC_FIELDS:
            sim = semantic_similarity(client, embed_model,
                                      str(gen_val or ""), str(gt_val or ""))
            ok = sim >= sim_threshold
            fields[key] = {"type": "semantic", "sim": round(sim, 3), "match": ok}
        else:
            ok = norm_value(gen_val) == norm_value(gt_val)
            fields[key] = {"type": "exact", "match": ok}
        all_ok = all_ok and ok
    return {"match": all_ok, "fields": fields, "checked": checked}


def score_match(client, embed_model: str, sim_threshold: float,
                expected: str, predicted: str) -> tuple[str, dict]:
    """
    Score a single ACP example structurally.

    Both `expected` and `predicted` are the model's JSON target format here
    (expected comes from serialize_acp_target; predicted from the model).

    Levels (best -> worst):
        exact            – identical JSON strings
        field            – decision + action + required-args all match
        decision_action  – decision + action match, resource differs
        wrong            – decision or action differs
        empty            – both expected & predicted are the deny-and-nothing {}
    """
    details: dict[str, Any] = {}
    exp_raw = (expected or "").strip()
    pred_raw = (predicted or "").strip()

    if exp_raw == pred_raw and exp_raw not in ("", "{}") and exp_raw.upper() != "NONE":
        return "exact", details

    exp = parse_prediction(exp_raw)
    pred = parse_prediction(pred_raw)
    details["expected_parsed"] = exp
    details["predicted_parsed"] = pred

    # Deny-and-do-nothing case: empty gold. Correct iff prediction is also empty.
    if not exp:
        return ("empty" if not pred else "wrong"), details

    exp_dec = str(exp.get("decision", "")).strip().lower()
    pred_dec = str(pred.get("decision", "")).strip().lower()
    exp_act = str(exp.get("action", "")).strip()
    pred_act = str(pred.get("action", "")).strip()

    res = compare_resource(client, embed_model, sim_threshold,
                           pred_act, exp_act,
                           pred.get("resource", {}), exp.get("resource", {}))

    details["decision_match"] = exp_dec == pred_dec
    details["action_match"] = action_matches(pred_act, exp_act)
    details["resource_match"] = res["match"]
    details["resource_fields"] = res["fields"]
    details["resource_checked"] = res["checked"]

    if details["decision_match"] and details["action_match"] and res["match"]:
        return "field", details
    if details["decision_match"] and details["action_match"]:
        return "decision_action", details
    return "wrong", details