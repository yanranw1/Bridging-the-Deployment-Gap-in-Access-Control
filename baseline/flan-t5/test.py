"""
NL → CODE Test Script
=====================
Loads a test JSON file (same schema as training: each example has NL / CODE /
ACP), runs the trained FLAN-T5 model on the serialized NL conversation, then
scores the generated CODE against the gold CODE.

Consistent with train.py:
    * input  = serialize_nl(ex["NL"])         (ACP is never fed to the model)
    * target = ex["CODE"]  -> JSON list of objects, each with fields
               {acp_id, api, args, decision, reason}

Matching strategy (best -> worst), per CODE object, then aggregated:
    exact            - generated CODE JSON == gold CODE JSON (normalized)
    field            - decision + api + required args all match
    decision_action  - decision + api match, args differ
    wrong            - decision or api differs / unparseable

Usage:
    python new_test.py \
        --model_dir ./outputs/flan_t5_nl2code \
        --test_path combined_no0_test.json \
        --output_path results.csv
"""

import argparse
import csv
import json
import logging
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Reuse the EXACT serialization used at training time so inputs match.
from train import serialize_nl, serialize_code, MAX_INPUT_LEN, MAX_TARGET_LEN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Required args keyed by api name (mirrors the CODE "args" object).
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

# Free-text args scored by SEMANTIC similarity (>= threshold = match).
# Everything else is scored by exact (normalized) equality.
SEMANTIC_FIELDS = {
    "query", "tone", "instructions", "subject",
    "body", "focus", "description", "message",
}

SIM_THRESHOLD = 0.75

# api aliases: apis acceptable as substitutes for a ground-truth api.
# Keyed by the GROUND-TRUTH api; every api implicitly matches itself.
# Asymmetric by design (send_email can stand in for reply/forward).
_API_ALIASES: dict[str, set[str]] = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
    "send_email":    {"send_email",    "forward_email"},
}


# ---------------------------------------------------------------------------
# Data loading (matches train.py schema, but keeps raw NL/CODE for scoring)
# ---------------------------------------------------------------------------

def load_test_examples(path: str) -> list[dict]:
    """Load test examples in the training schema.

    Returns a list of dicts with:
        input_text : serialized NL prompt (same as training input)
        nl_turns   : raw NL turns (for re-serialization / debugging)
        code_gold  : the gold CODE list (target)
    ACP is intentionally ignored.
    """
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    examples = []
    for ex in data:
        examples.append({
            "input_text": serialize_nl(ex["NL"]),
            "nl_turns":   ex["NL"],
            "code_gold":  ex["CODE"],
        })
    return examples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _load_model(model_dir: str):
    if model_dir not in _MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        mdl.eval()
        if torch.cuda.is_available():
            mdl = mdl.cuda()
        _MODEL_CACHE[model_dir] = (tok, mdl)
    return _MODEL_CACHE[model_dir]


def predict(model_dir: str, input_text: str) -> str:
    """Run the trained model on a single serialized NL prompt.
    Returns the raw decoded string (expected to be a CODE JSON list)."""
    tok, mdl = _load_model(model_dir)
    enc = tok(input_text, max_length=MAX_INPUT_LEN, truncation=True,
              return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl.generate(**enc, max_length=MAX_TARGET_LEN)
    return tok.decode(out[0], skip_special_tokens=True)


def predict_base_model(model_dir: str | None, nl_turns: list[dict]) -> str:
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    prompt = "translate to ACP: " + format_nl_conversation(nl_turns)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# CODE parsing / normalization
# ---------------------------------------------------------------------------

_CODE_KEYS = ["acp_id", "api", "args", "decision", "reason"]


def repair_code_json(s: str) -> str:
    """Repair the common seq2seq 'debrace' failure where the model emits
    valid-ish CODE but drops the '{' '}' around (a) each list element and
    (b) the args object, e.g.:

        ["acp_id":"a1","api":"get_email","args":"email_id":"x", ...]

    Conservative by design: only called after a normal json.loads fails, and
    only re-wraps using the known CODE keys, so it won't corrupt valid output.
    """
    s = s.strip()

    # 1. Wrap a bare args value in braces: "args": <pairs> ,"<next known key>"
    next_key = "|".join(k for k in _CODE_KEYS if k != "args")

    def _wrap_args(m):
        inner = m.group(1).strip()
        if inner.startswith("{") or not inner:
            return m.group(0)
        return f'"args":{{{inner}}}'

    s = re.sub(r'"args":\s*(.*?)(?=,\s*"(?:%s)"\s*:)' % next_key,
               _wrap_args, s, flags=re.DOTALL)

    # 2. Wrap top-level list elements in braces (elements start at "acp_id").
    if s.startswith("[") and not re.match(r'\[\s*\{', s):
        body = s[1:-1] if s.endswith("]") else s[1:]
        elems = re.split(r'(?<=")\s*,\s*(?="acp_id"\s*:)', body)
        s = "[" + ",".join("{" + e.strip().strip(",") + "}" for e in elems) + "]"

    return s


def parse_code(text) -> list[dict]:
    """Parse a CODE payload into a list of code-object dicts.
    Accepts an already-parsed list, a JSON string, or a JSON object.
    Returns [] if it can't be parsed into a list of dicts."""
    if isinstance(text, list):
        return [c for c in text if isinstance(c, dict)]
    if isinstance(text, dict):
        return [text]
    s = (text or "").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Repair the common debrace failure (missing { } on elements / args).
        try:
            obj = json.loads(repair_code_json(s))
        except json.JSONDecodeError:
            # Best-effort: pull the first {...} or [...] block out of noisy output.
            m = re.search(r'(\[.*\]|\{.*\})', s, re.DOTALL)
            if not m:
                return []
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict)]
    return []


def norm_value(val) -> object:
    """Normalize a scalar value for exact comparison.
    Strings: lower-cased + whitespace-collapsed. Numbers: compared by value."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"\s+", " ", str(val).strip().lower())
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return s


def _api_normalize(api: str) -> str:
    """Strip underscores/spaces and lowercase for loose comparison."""
    return re.sub(r"[\s_]", "", api or "").lower()


def api_matches(exp_api: str, pred_api: str) -> bool:
    """True if pred_api is an acceptable match for exp_api (exact or alias)."""
    exp_norm = _api_normalize(exp_api)
    pred_norm = _api_normalize(pred_api)
    if exp_norm == pred_norm:
        return True
    for canonical, aliases in _API_ALIASES.items():
        if exp_norm == _api_normalize(canonical):
            return pred_norm in {_api_normalize(a) for a in aliases}
    return False


def _api_to_canon(api: str) -> str:
    """Map a (possibly spaced/cased) api name to a REQUIRED_ARGS key."""
    raw = (api or "").strip().lower()
    snake = re.sub(r"\s+", "_", raw)
    if snake in REQUIRED_ARGS:
        return snake
    if raw in REQUIRED_ARGS:
        return raw
    return snake


# ---------------------------------------------------------------------------
# Semantic similarity (for free-text args)
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
    """Cosine similarity of embeddings; difflib fallback. Returns [0, 1]."""
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
# Match scoring (operates on CODE args dicts)
# ---------------------------------------------------------------------------

def compare_args(client, embed_model: str, gen_api: str, gt_api: str,
                 gen_args: dict, gt_args: dict) -> dict:
    """Compare the REQUIRED args of the GENERATED api.

    Schema is driven by the generated api (important when an alias was used).
    We validate every required key of gen_api for which ground truth provides
    a value. Free-text fields use semantic similarity; others use exact
    normalized equality. Optional args are ignored.
    """
    gen_args = gen_args or {}
    gt_args = gt_args or {}
    print("gen_api",gen_api)
    print("gt_api",gt_api)


    if gen_api in REQUIRED_ARGS:
        required = REQUIRED_ARGS[gen_api]
    elif gt_api in REQUIRED_ARGS:
        required = REQUIRED_ARGS[gt_api]
    else:
        required = list(gt_args.keys())

    field_results = {}
    all_ok = True
    # checked_any = False
    for key in required:
        if key not in gt_args:
            field_results[key] = {"type": "skipped", "match": None,
                                  "note": "no ground-truth value"}
            continue
        #checked_any = True
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

    return {"match": all_ok, "fields": field_results}#"checked": checked_any}


def _score_single(client, embed_model, exp_obj: dict, pred_obj: dict) -> tuple[str, dict]:
    """Score one gold CODE object against one predicted CODE object."""
    details = {}

    exp_decision = str(exp_obj.get("decision", "")).strip().lower()
    pred_decision = str(pred_obj.get("decision", "")).strip().lower()
    exp_api = str(exp_obj.get("api", "")).strip()
    pred_api = str(pred_obj.get("api", "")).strip()

    exp_args = exp_obj.get("args", {}) or {}
    pred_args = pred_obj.get("args", {}) or {}

    res = compare_args(client, embed_model,
                       _api_to_canon(pred_api), _api_to_canon(exp_api),
                       pred_args, exp_args)

    details["decision_match"] = exp_decision == pred_decision
    details["api_match"] = api_matches(pred_api, exp_api)
    details["args_match"] = res["match"]
    details["arg_fields"] = res["fields"]
    # details["args_checked"] = res["checked"]

    if all([details["decision_match"], details["api_match"], details["args_match"]]):
        return "field", details
    if details["decision_match"] and details["api_match"]:
        return "decision_action", details
    return "wrong", details


_LEVEL_RANK = {"exact": 0, "field": 1, "decision_action": 2, "wrong": 3}


def _sorted(obj: dict) -> dict:
    """Normalize a CODE object for exact comparison: lowercase keys,
    normalize scalar values, normalize args."""
    out = {}
    for k, v in (obj or {}).items():
        if isinstance(v, dict):
            out[k.lower()] = {kk.lower(): norm_value(vv) for kk, vv in v.items()}
        else:
            out[k.lower()] = norm_value(v)
    return out


def score_match(client, embed_model, expected, predicted) -> tuple[str, dict]:
    """
    Score the predicted CODE list against the gold CODE list.

    Levels (best -> worst):
        'exact'           - normalized CODE JSON identical
        'field'           - decision + api + required args match (per object)
        'decision_action' - decision + api match, args differ
        'wrong'           - decision or api differs / unparseable / count mismatch

    Multi-object CODE: each gold object is matched positionally to the
    predicted object; the OVERALL level is the WORST per-object level.
    """
    exp_list = parse_code(expected)
    pred_list = parse_code(predicted)

    details = {"expected_parsed": exp_list, "predicted_parsed": pred_list}

    # 1. Exact (normalized re-serialization, key-order independent)
    try:
        exp_norm = json.dumps([_sorted(o) for o in exp_list], sort_keys=True)
        pred_norm = json.dumps([_sorted(o) for o in pred_list], sort_keys=True)
        if exp_list and exp_norm == pred_norm:
            return "exact", details
    except (TypeError, ValueError):
        pass

    # if not pred_list:
    #     print(1)
    #     details[""] = "could not parse predicted CODE"
    #     return "wrong", details

    if len(exp_list) != len(pred_list):
        details["count_mismatch"] = {"expected": len(exp_list),
                                     "predicted": len(pred_list)}

    # 2/3/4. Per-object scoring; overall = worst level across the gold objects.
    per_object = []
    worst = "field"
    n = max(len(exp_list), len(pred_list))
    for i in range(n):
        exp_obj = exp_list[i] if i < len(exp_list) else {}
        pred_obj = pred_list[i] if i < len(pred_list) else {}
        lvl, d = _score_single(client, embed_model, exp_obj, pred_obj)
        per_object.append({"index": i, "level": lvl, **d})
        if _LEVEL_RANK[lvl] > _LEVEL_RANK[worst]:
            worst = lvl

    # A count mismatch can never be better than decision_action.
    if "count_mismatch" in details and _LEVEL_RANK[worst] < _LEVEL_RANK["decision_action"]:
        worst = "decision_action"

    details["per_object"] = per_object
    return worst, details


MATCH_ICON = {
    "exact":           "OK OK",
    "field":           "OK ",
    "decision_action": "~  ",
    "wrong":           "X  ",
}


# ---------------------------------------------------------------------------
# Core test loop
# ---------------------------------------------------------------------------

def run_tests(model_dir: str, test_path: str, client, embed_model: str) -> list[dict]:
    examples = load_test_examples(test_path)
    results = []

    for i, ex in enumerate(examples):
        input_text = ex["input_text"]
        expected = ex["code_gold"]

        logger.info("Running example %d / %d ...", i + 1, len(examples))

        predicted_str = predict_base_model(model_dir, input_text)
        expected_str = serialize_code(expected)

        level, details = score_match(client, embed_model, expected, predicted_str)
        icon = MATCH_ICON[level]

        result = {
            "index":      i,
            "input":      input_text,
            "expected":   expected_str,
            "predicted":  predicted_str,
            "match":      level,          # exact / field / decision_action / wrong
            "exact":      level == "exact",
            "field_match": level in ("exact", "field"),
            "decision_action_match": level in ("exact", "field", "decision_action"),
            "details":    details,
        }
        results.append(result)

        print(f"\n{'='*70}")
        print(f"[{i}] INPUT:\n{input_text}")
        print(f"\n  EXPECTED : {expected_str}")
        print(f"  PREDICTED: {predicted_str}")
        print(f"  MATCH    : {icon} ({level})")
        if level not in ("exact", "field") and details.get("per_object"):
            for po in details["per_object"]:
                flags = {k: v for k, v in po.items() if k.endswith("_match")}
                print(f"  OBJ[{po['index']}] {po['level']}: {flags}")

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
    logger.info("CSV saved -> %s", path)


def save_results_json(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("JSON saved -> %s", path)



def print_summary(results: list[dict]) -> None:
    total = len(results) or 1
    exact = sum(1 for r in results if r["match"] == "exact")
    field = sum(1 for r in results if r["match"] == "field")
    dec_act = sum(1 for r in results if r["match"] == "decision_action")
    wrong = sum(1 for r in results if r["match"] == "wrong")
        
    def _example_flag(r, flag):
        if r["match"] == "exact":
            return True
        return all(po.get(flag) for po in r["details"].get("per_object", []))

    api_match      = sum(1 for r in results if _example_flag(r, "api_match"))
    decision_match = sum(1 for r in results if _example_flag(r, "decision_match"))
    args_match     = sum(1 for r in results if _example_flag(r, "args_match"))
    new_exact = sum(1 for r in results if _example_flag(r, "api_match") and _example_flag(r, "decision_match") and _example_flag(r, "args_match"))
    for r in results:
        if r["match"] in ("exact","field") and not _example_flag(r, "api_match"):
            print(r["index"], r["match"], r["details"].get("per_object"))
    print(f"\n{'='*70}")
    print(f"RESULTS ({len(results)} examples)")
    print(f"  Exact match          : {exact:3d}  ({100*exact/total:.1f}%)")
    print(f"  Field match          : {field:3d}  ({100*field/total:.1f}%)")
    print(f"  Decision+API only    : {dec_act:3d}  ({100*dec_act/total:.1f}%)")
    print(f"  Wrong                : {wrong:3d}  ({100*wrong/total:.1f}%)")
    print(f"  decision_match       : {decision_match:3d}  ({100*decision_match/total:.1f}%)")
    print(f"  api_match            : {api_match:3d}  ({100*api_match/total:.1f}%)")
    print(f"  args_match           : {args_match:3d}  ({100*args_match/total:.1f}%)")
    print(f"  new_exact           : {new_exact:3d}  ({100*new_exact/total:.1f}%)")

    print(f"  Correct (>=field)    : {exact+field:3d}  ({100*(exact+field)/total:.1f}%)")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    client = None  # plug in an OpenAI client here for embedding-based matching
    results = run_tests(args.model_dir, args.test_path, client, args.embed_model)
    print_summary(results)

    save_results_csv(results, args.output_path)

    json_path = args.output_path.replace(".csv", ".json")
    if json_path == args.output_path:
        json_path += ".json"
    save_results_json(results, json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained NL -> CODE model.")

    parser.add_argument("--model_dir",   type=str, default="./outputs/flan_t5_nl2code",
                        help="Path to the saved model directory")
    parser.add_argument("--test_path",   type=str,
                        default="/home/ubuntu/agentv-main/email_agent/dataset/combined_with0_test.json",
                        help="Test JSON file (same schema as training data)")
    parser.add_argument("--output_path", type=str, default="results.csv",
                        help="Where to save results (CSV + JSON written alongside)")
    parser.add_argument("--embed-model", dest="embed_model", type=str,
                        default="text-embedding-3-small",
                        help="Embedding model for semantic field matching")

    args = parser.parse_args()
    main(args)