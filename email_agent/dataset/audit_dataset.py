import json
import re
from pathlib import Path
from collections import OrderedDict


# ============================================================
# Dispatch contract: allowed APIs and their arguments
# ============================================================


API_CONTRACT = {
    "search_emails": {"required": ["query"], "optional": ["max_results"],
                      "method": "users.messages.list", "returns": "emails[]"},
    "list_emails":   {"required": [], "optional": ["label", "max_results"],
                      "method": "users.messages.list", "returns": "emails[]"},
    "get_email":     {"required": ["email_id"], "optional": [],
                      "method": "users.messages.get", "returns": "email"},
    "analyze_email": {"required": ["email_id"], "optional": ["aspects"],
                      "method": "(local LLM analysis)", "returns": "analysis"},
    "send_email":    {"required": ["to", "subject", "body"], "optional": ["cc", "bcc"],
                      "method": "users.messages.send", "returns": "message",
                      "destructive": True},
    "draft_email":   {"required": ["to", "subject", "body"], "optional": ["cc"],
                      "method": "users.drafts.create", "returns": "draft"},
    "draft_reply":   {"required": ["email_id", "body"], "optional": [],
                      "method": "users.drafts.create", "returns": "draft"},
    "forward_email": {"required": ["email_id", "to"], "optional": ["message"],
                      "method": "users.messages.send", "returns": "message",
                      "destructive": True},
    "star_email":    {"required": ["email_id"], "optional": [],
                      "method": "users.messages.modify", "returns": "message"},
    "send_draft":    {"required": ["draft_id"], "optional": [],
                      "method": "users.drafts.send", "returns": "message",
                      "destructive": True},
    "delete_email":  {"required": ["email_id"], "optional": [],
                      "method": "users.messages.trash", "returns": "message",
                      "destructive": True},
    "create_task":   {"required": ["title"], "optional": ["due", "notes"],
                      "method": "tasks.insert", "returns": "task"},
    "complete_task": {"required": ["task_id"], "optional": [],
                      "method": "tasks.patch", "returns": "task"},
    "list_tasks":    {"required": [], "optional": ["max_results"],
                      "method": "tasks.list", "returns": "tasks[]"},
}


# ============================================================
# ACP resource parser
#   "email_id: msg1, to: a@b.com" -> {"email_id": "msg1", "to": "a@b.com"}
# ============================================================

KEY = r"[A-Za-z_][A-Za-z0-9_]*"
PAIR = re.compile(rf"^({KEY}): (.*)$")


def parse_resource(resource):
    """Parse an ACP resource string into a dict of key -> value (str)."""
    if not isinstance(resource, str) or resource.strip() == "":
        return {}
    parts = re.split(rf", (?={KEY}: )", resource)
    parsed = {}
    for part in parts:
        m = PAIR.match(part.strip())
        if m:
            parsed[m.group(1)] = m.group(2)
    return parsed


def norm(value):
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def canon(value):
    """Canonicalize a value for comparison so that equivalent ACP/CODE
    forms compare equal: unify acp_N_return / code_N_return reference
    placeholders, and strip wrapping quotes."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1]
    # <acp_0_return.id> and <code_0_return.id> refer to the same value
    v = re.sub(r"\b(?:acp|code)(_\d+_return)", r"ref\1", v)
    return v


# ============================================================
# Audit: does each ACP entry match its paired CODE entry?
# Pairing is by acp_id (ACP) == code_id (CODE).
# ============================================================

def audit_item(item, item_index, file_path):
    rows = []
    acp_list = item.get("ACP", [])
    code_list = item.get("CODE", [])
    code_by_id = {c.get("code_id"): c for c in code_list if isinstance(c, dict)}

    for acp in acp_list:
        if not isinstance(acp, dict):
            continue
        problems = []
        acp_id = acp.get("acp_id")
        code = code_by_id.get(acp_id)

        if code is None:
            rows.append(_row(file_path, item_index, item, acp_id,
                             [f"No CODE entry with code_id matching acp_id '{acp_id}'"]))
            continue

        # action vs api
        action, api = acp.get("action"), code.get("api")
        if action != api:
            problems.append(f"action/api mismatch: ACP action='{action}', CODE api='{api}'")

        # decision
        if acp.get("decision") != code.get("decision"):
            problems.append(
                f"decision mismatch: ACP='{acp.get('decision')}', CODE='{code.get('decision')}'"
            )

        # resource args vs code args
        acp_args = parse_resource(acp.get("resource"))
        code_args = code.get("args", {})
        if not isinstance(code_args, dict):
            problems.append(f"CODE args must be an object, got {type(code_args).__name__}")
            code_args = {}

        # validate against the dispatch contract
        api_name = api or action
        if api_name not in API_CONTRACT:
            problems.append(f"Unknown API/action '{api_name}' not in dispatch contract")
        else:
            contract = API_CONTRACT[api_name]
            required = set(contract["required"])
            allowed = required | set(contract["optional"])

            for label, keys in (("ACP resource", set(acp_args)), ("CODE args", set(code_args))):
                missing = required - keys
                if missing:
                    problems.append(f"Missing required arg(s) in {label}: {sorted(missing)}")
                unsupported = keys - allowed
                if unsupported:
                    problems.append(
                        f"Unsupported arg(s) in {label} for {api_name}: "
                        f"{sorted(unsupported)}; allowed={sorted(allowed)}"
                    )

        if set(acp_args) != set(code_args):
            problems.append(
                f"arg keys differ: ACP={sorted(acp_args)}, CODE={sorted(code_args)}"
            )

        for key in set(acp_args) & set(code_args):
            av, cv = acp_args[key], norm(code_args[key])
            if {av.lower(), cv.lower()} <= {"true", "false"}:
                ok = av.lower() == cv.lower()
            else:
                ok = canon(av) == canon(cv)
            if not ok:
                problems.append(f"value mismatch for '{key}': ACP='{av}', CODE='{cv}'")

        if problems:
            rows.append(_row(file_path, item_index, item, acp_id, problems))

    return rows


def _row(file_path, item_index, item, acp_id, problems):
    return {
        "file": str(file_path),
        "item_index": item_index,
        "class": item.get("class"),
        "id": item.get("id"),
        "acp_id": acp_id,
        "problems": problems,
    }


# ============================================================
# Loading
# ============================================================

def find_items(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Expected a list or a dict with data/examples/items.")


def audit_path(input_path):
    path = Path(input_path)
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    rows = []
    for file_path in files:
        try:
            items = find_items(json.load(open(file_path, encoding="utf-8")))
        except Exception as e:
            rows.append(_row(file_path, None, {}, None, [f"Failed to load JSON: {e}"]))
            continue
        for i, item in enumerate(items):
            if isinstance(item, dict):
                rows.extend(audit_item(item, i, file_path))
    return rows


# ============================================================
# Output
# ============================================================

def print_summary(rows):
    if not rows:
        print("✅ No issues found.")
        return
    print(f"❌ Found {len(rows)} mismatched ACP/CODE pair(s).\n")
    for row in rows:
        print(f"id={row['id']} | class={row['class']} | acp_id={row['acp_id']}")
        for i, p in enumerate(row["problems"], 1):
            print(f"  {i}. {p}")
        print("-" * 80)


def main():
    input_path = "combined.json"
    rows = audit_path(input_path)
    print_summary(rows)
    json.dump(rows, open("audit_report.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nJSON audit saved to: audit_report.json")


if __name__ == "__main__":
    main()