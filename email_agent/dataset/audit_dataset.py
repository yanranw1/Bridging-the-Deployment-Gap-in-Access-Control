import json
import csv
import re
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict


# ============================================================
# Dispatch contract, in the same order as the code
# ============================================================

API_CONTRACT = OrderedDict([
    ("list_emails", {
        "required": [],
        "optional": ["max_results", "query"],
    }),
    ("get_email", {
        "required": ["email_id"],
        "optional": [],
    }),
    ("search_emails", {
        "required": ["query"],
        "optional": ["max_results"],
    }),
    ("analyze_email", {
        "required": ["email_id"],
        "optional": [],
    }),
    ("draft_reply", {
        "required": ["email_id"],
        "optional": ["tone", "instructions"],
    }),
    ("send_email", {
        "required": ["to", "subject", "body"],
        "optional": ["reply_to_id", "thread_id"],
    }),
    ("send_draft", {
        "required": ["draft_id"],
        "optional": [],
    }),
    ("summarize_inbox", {
        "required": [],
        "optional": ["focus", "max_results"],
    }),
    ("create_task", {
        "required": ["title"],
        "optional": ["description", "deadline", "email_id", "email_subject", "priority"],
    }),
    ("list_tasks", {
        "required": [],
        "optional": ["status"],
    }),
    ("complete_task", {
        "required": ["task_id"],
        "optional": [],
    }),
    ("forward_email", {
        "required": ["email_id", "to"],
        "optional": ["message"],
    }),
    ("delete_email", {
        "required": ["email_id"],
        "optional": [],
    }),
    ("draft_email", {
        "required": ["body"],
        "optional": ["to", "subject"],
    }),
    ("star_email", {
        "required": ["email_id"],
        "optional": ["star"],
    }),
])


# ============================================================
# ACP resource parser
# Required format:
#   "title: Finish report"
#   "title: Finish report, description: Submit by Friday"
#
# Bad:
#   "Finish report"
#   {"title": "Finish report"}
#   "title: Finish report,description: Submit by Friday"
# ============================================================

KEY_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
PAIR_PATTERN = re.compile(rf"^({KEY_PATTERN}): (.*)$")


def normalize_value(value):
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def split_resource_pairs(resource):
    """
    Split on comma + space only when it is followed by key:
    Example:
      "email_id: msg1, to: a@b.com, message: Hi, can you check this?"
    becomes:
      ["email_id: msg1", "to: a@b.com", "message: Hi, can you check this?"]
    """
    return re.split(rf", (?={KEY_PATTERN}: )", resource)


def parse_acp_resource(resource):
    """
    Returns:
      parsed_args: dict
      errors: list[str]
    """
    errors = []

    if resource is None:
        return {}, []

    if not isinstance(resource, str):
        return {}, [f"ACP resource must be a string, but got {type(resource).__name__}"]

    if resource.strip() == "":
        return {}, []

    # Detect wrong separator like ",description: x"
    if re.search(rf",(?={KEY_PATTERN}: )", resource):
        errors.append("ACP resource uses bad separator. Use comma + space: ', '")

    parts = split_resource_pairs(resource)
    parsed = {}

    for part in parts:
        part = part.strip()
        match = PAIR_PATTERN.match(part)

        if not match:
            errors.append(f"Bad ACP resource pair: '{part}'. Expected format 'key: value'")
            continue

        key, value = match.group(1), match.group(2)

        if key in parsed:
            errors.append(f"Duplicate key in ACP resource: '{key}'")

        parsed[key] = value

    return parsed, errors


# ============================================================
# JSON loading
# ============================================================

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_dataset_items(data):
    """
    Supports:
      [ {...}, {...} ]
      { "data": [ ... ] }
      { "examples": [ ... ] }
      { "items": [ ... ] }
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "examples", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unsupported JSON shape. Expected list or dict with data/examples/items list.")


def get_json_files(input_path):
    path = Path(input_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(path.rglob("*.json"))

    raise FileNotFoundError(f"Path not found: {input_path}")


def get_code_by_acp_id(code_list):
    result = {}

    if not isinstance(code_list, list):
        return result

    for code in code_list:
        if isinstance(code, dict):
            acp_id = code.get("acp_id")
            if acp_id:
                result[acp_id] = code

    return result


# ============================================================
# Audit logic
# ============================================================

def make_audit_row(file_path, item_index, item, acp_id, problems):
    """
    One row can include multiple problems together.
    """
    return {
        "file": str(file_path),
        "item_index": item_index,
        "class": item.get("class"),
        "id": item.get("id"),
        "acp_id": acp_id,
        "problem_count": len(problems),
        "problems": " | ".join(problems),
    }


def audit_item(item, item_index, file_path):
    audit_rows = []

    acp_list = item.get("ACP", [])
    code_list = item.get("CODE", [])

    if not isinstance(acp_list, list):
        return [make_audit_row(
            file_path,
            item_index,
            item,
            None,
            [f"ACP must be a list, but got {type(acp_list).__name__}"]
        )]

    if not isinstance(code_list, list):
        return [make_audit_row(
            file_path,
            item_index,
            item,
            None,
            [f"CODE must be a list, but got {type(code_list).__name__}"]
        )]

    code_by_acp_id = get_code_by_acp_id(code_list)

    for acp in acp_list:
        problems = []

        if not isinstance(acp, dict):
            audit_rows.append(make_audit_row(
                file_path,
                item_index,
                item,
                None,
                [f"ACP entry must be an object, but got {type(acp).__name__}"]
            ))
            continue

        acp_id = acp.get("acp_id")
        action = acp.get("action")
        resource = acp.get("resource")

        if not acp_id:
            audit_rows.append(make_audit_row(
                file_path,
                item_index,
                item,
                None,
                ["Missing ACP acp_id"]
            ))
            continue

        code = code_by_acp_id.get(acp_id)

        if not code:
            audit_rows.append(make_audit_row(
                file_path,
                item_index,
                item,
                acp_id,
                [f"Missing matching CODE entry for acp_id '{acp_id}'"]
            ))
            continue

        api = code.get("api")
        args = code.get("args", {})

        if not isinstance(args, dict):
            problems.append(f"CODE args must be an object, but got {type(args).__name__}")
            args = {}

        # ----------------------------------------------------
        # Rule 3: ACP action must match CODE api
        # ----------------------------------------------------
        if action != api:
            problems.append(f"ACP action does not match CODE api: ACP action='{action}', CODE api='{api}'")

        api_name = api or action

        if api_name not in API_CONTRACT:
            problems.append(f"Unknown API/action '{api_name}' not found in dispatch contract")
            audit_rows.append(make_audit_row(
                file_path,
                item_index,
                item,
                acp_id,
                problems
            ))
            continue

        contract = API_CONTRACT[api_name]
        required = set(contract["required"])
        optional = set(contract["optional"])
        allowed = required | optional

        # ----------------------------------------------------
        # Rule 2: ACP resource format
        # ----------------------------------------------------
        acp_resource_args, resource_format_errors = parse_acp_resource(resource)

        for error in resource_format_errors:
            problems.append(error)

        acp_keys = set(acp_resource_args.keys())
        code_keys = set(args.keys())

        # ----------------------------------------------------
        # Rule 1: Required arguments
        # ----------------------------------------------------
        missing_required_in_acp = required - acp_keys
        if missing_required_in_acp:
            problems.append(
                f"Missing required argument(s) in ACP resource: {sorted(missing_required_in_acp)}"
            )

        missing_required_in_code = required - code_keys
        if missing_required_in_code:
            problems.append(
                f"Missing required argument(s) in CODE args: {sorted(missing_required_in_code)}"
            )

        # ----------------------------------------------------
        # Rule 1: Unsupported arguments
        # ----------------------------------------------------
        unsupported_acp_args = acp_keys - allowed
        if unsupported_acp_args:
            problems.append(
                f"Unsupported argument(s) in ACP resource for {api_name}: {sorted(unsupported_acp_args)}; allowed={sorted(allowed)}"
            )

        unsupported_code_args = code_keys - allowed
        if unsupported_code_args:
            problems.append(
                f"Unsupported argument(s) in CODE args for {api_name}: {sorted(unsupported_code_args)}; allowed={sorted(allowed)}"
            )

        # ----------------------------------------------------
        # Rule 3: ACP resource keys must match CODE args keys
        # ----------------------------------------------------
        if acp_keys != code_keys:
            problems.append(
                f"ACP resource keys do not match CODE args keys: ACP keys={sorted(acp_keys)}, CODE keys={sorted(code_keys)}"
            )

        # ----------------------------------------------------
        # Rule 3: ACP resource values must match CODE args values
        # ----------------------------------------------------
        shared_keys = acp_keys & code_keys

        for key in sorted(shared_keys):
            acp_value = acp_resource_args[key]
            code_value = normalize_value(args[key])

            if acp_value.lower() in ["true", "false"] and code_value.lower() in ["true", "false"]:
                match = acp_value.lower() == code_value.lower()
            else:
                match = acp_value == code_value

            if not match:
                problems.append(
                    f"Value mismatch for '{key}': ACP has '{acp_value}', CODE has '{code_value}'"
                )

        if problems:
            audit_rows.append(make_audit_row(
                file_path,
                item_index,
                item,
                acp_id,
                problems
            ))

    return audit_rows


def audit_path(input_path):
    all_rows = []
    files = get_json_files(input_path)

    for file_path in files:
        try:
            data = load_json_file(file_path)
            items = find_dataset_items(data)
        except Exception as e:
            all_rows.append({
                "file": str(file_path),
                "item_index": None,
                "class": None,
                "id": None,
                "acp_id": None,
                "problem_count": 1,
                "problems": f"Failed to load or parse JSON file: {e}",
            })
            continue

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                all_rows.append({
                    "file": str(file_path),
                    "item_index": i,
                    "class": None,
                    "id": None,
                    "acp_id": None,
                    "problem_count": 1,
                    "problems": f"Dataset item must be object, but got {type(item).__name__}",
                })
                continue

            all_rows.extend(audit_item(item, i, file_path))

    return all_rows


# ============================================================
# Output
# ============================================================

def write_csv(rows, output_csv):
    fieldnames = [
        "file",
        "item_index",
        "class",
        "id",
        "acp_id",
        "problem_count",
        "problems",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows, output_json):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def print_summary(rows):
    if not rows:
        print("✅ No issues found.")
        return

    print(f"❌ Found {len(rows)} problematic ACP item(s).\n")

    for row in rows:
        print(
            f"file={row['file']} | "
            f"class={row['class']} | "
            f"id={row['id']} | "
            f"acp_id={row['acp_id']} | "
            f"problem_count={row['problem_count']}"
        )

        problems = row["problems"].split(" | ")
        for idx, problem in enumerate(problems, start=1):
            print(f"  {idx}. {problem}")

        print("-" * 100)


def main():
    input_path = "/Users/yanran/Downloads/cleaned/class6.json"
    csv = "audit_report.csv"
    json = "audit_report.json"


    rows = audit_path(input_path)

    print_summary(rows)
    write_csv(rows, csv)
    write_json(rows, json)

    print(f"\nCSV audit saved to: {csv}")
    print(f"JSON audit saved to: {json}")


if __name__ == "__main__":
    main()