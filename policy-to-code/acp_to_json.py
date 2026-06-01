"""
ACP Text → JSON Converter
=========================
Converts a flat ACP string of the form

    {decision: deny; subject: email_agent; action: forward_email;
     resource: email_id=msg_..., to: ..., message: ...;
     purpose: ...; condition: ...}

into the structured JSON list format:

    [
      {
        "acp_id": "acp_1",
        "decision": "deny",
        "subject": "email_agent",
        "action": "forward_email",
        "resource": "email_id: msg_..., to: ..., message: ...",
        "purpose": "...",
        "condition": "...",
        "policy_metadata": { ... }
      }
    ]

Usage:
    python acp_to_json.py --text "{decision: deny; ...}"
    python acp_to_json.py --text "..." --output out.json
    echo "{decision: ...}" | python acp_to_json.py
"""

import argparse
import json
import re
import sys


# Top-level fields recognised at the "; key:" boundary level.
# `resource` is multi-part (contains its own commas/colons) and handled specially.
_TOP_LEVEL_FIELDS = ["decision", "subject", "action", "resource", "purpose", "condition"]


def parse_acp(text: str) -> dict[str, str]:
    """
    Parse a flat ACP string into a dict of field → value.

    Splits only on semicolons that are immediately followed by one of the
    known top-level field names, so commas/colons inside a value (e.g. inside
    `resource` or `message`) are preserved.
    """
    text = text.strip().lstrip("{").rstrip("}").strip()

    # Build an alternation of known field names for the split boundary.
    field_alt = "|".join(_TOP_LEVEL_FIELDS)
    parts = re.split(rf';\s*(?=(?:{field_alt})\s*:)', text)

    result: dict[str, str] = {}
    for part in parts:
        m = re.match(r'\s*(\w+)\s*:\s*(.*)', part.strip(), re.DOTALL)
        if m:
            result[m.group(1).lower().strip()] = m.group(2).strip()
    return result


def normalize_resource(resource_val: str) -> str:
    """
    Convert the resource value's leading `key=value` token into `key: value`
    so the output matches the target format
    (`email_id=msg_x, to: ...`  →  `email_id: msg_x, to: ...`).
    In the source data only the first token uses `=`; subsequent tokens
    already use `: `.
    """
    return re.sub(
        r'^(\w+)\s*=\s*',
        r'\1: ',
        resource_val.strip()
    )


def split_policies(text: str) -> list[str]:
    """
    Split a text blob that may contain multiple ACPs into individual policy
    strings. Policies are separated by a pipe `|` (optionally surrounded by
    whitespace). The outer `{ ... }` braces are stripped first so a single
    pair of braces wrapping several policies is handled correctly.
    """
    text = text.strip().lstrip("{").rstrip("}").strip()
    chunks = re.split(r'\s*\|\s*', text)
    return [c.strip() for c in chunks if c.strip()]


def _build_one(text: str, acp_id: str) -> dict:
    """Build a single ACP object from one policy string."""
    fields = parse_acp(text)

    # The CSV encodes "no condition" as the literal string "none"; the gold
    # JSON represents it as null. Normalise to None so they compare equal.
    condition = fields.get("condition", "")
    if condition.strip().lower() in ("none", ""):
        condition = None

    return {
        "acp_id":   acp_id,
        "decision": fields.get("decision", ""),
        "subject":  fields.get("subject", ""),
        "action":   fields.get("action", ""),
        "resource": normalize_resource(fields.get("resource", "")),
        "purpose":  fields.get("purpose", ""),
        "condition": condition,
    }


def convert(text: str, id_prefix: str = "acp_") -> list[dict]:
    """
    Convert an ACP text string (one or more policies separated by `|`) into
    the structured JSON list. Each policy gets an auto-incrementing acp_id
    (acp_1, acp_2, …).
    """
    policies = split_policies(text)
    return [
        _build_one(p, f"{id_prefix}{i}")
        for i, p in enumerate(policies, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert flat ACP text to structured JSON.")
    parser.add_argument("--text", type=str, default=None,
                        help="The ACP string. If omitted, reads from stdin.")
    parser.add_argument("--id_prefix", type=str, default="acp_",
                        help="Prefix for auto-incrementing ACP ids (default: 'acp_' → acp_1, acp_2…).")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write the JSON output.")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("No ACP text provided (use --text or pipe via stdin).")

    result = convert(text, id_prefix=args.id_prefix)
    out_str = json.dumps(result, ensure_ascii=False, indent=2)

    print(out_str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"\nSaved → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()