#!/usr/bin/env python3
"""Extract the CODE (executable steps) from each ACP record in a combined test file.

Usage:
    python extract_code.py input.json [-o output.json] [--py]

By default writes a JSON file mapping each record id to its CODE steps.
With --py it also emits a runnable Python stub per record (api calls in order).
"""
import argparse
import json
import sys
from pathlib import Path


def extract(records):
    """Return {id: [code_steps]} for every record that has a CODE block."""
    out = {}
    for r in records:
        rid = r.get("id", f"idx_{len(out)}")
        code = r.get("CODE")
        if code:
            out[rid] = code
    return out


def to_python_stub(rid, steps):
    """Render a record's CODE steps as a sequence of api() calls."""
    lines = [f"# === {rid} ==="]
    for step in steps:
        api = step.get("api", "unknown_api")
        args = step.get("args", {})
        decision = step.get("decision", "")
        kw = ", ".join(f"{k}={v!r}" for k, v in args.items())
        prefix = "" if decision == "allow" else f"# [{decision}] "
        lines.append(f"{prefix}{api}({kw})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Extract CODE blocks from ACP records.")
    ap.add_argument("--input", type=Path ,default="/home/ubuntu/agentv-main/email_agent/dataset/combined_test.json")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output JSON path (default: <input>_code.json)")
    ap.add_argument("--py", action="store_true",
                    help="also write a .py stub of the api calls")
    args = ap.parse_args()

    records = json.loads(args.input.read_text())
    if isinstance(records, dict):
        records = [records]

    code_map = extract(records)
    out = args.output or args.input.with_name(args.input.stem + "_code.json")
    out.write_text(json.dumps(code_map, indent=2))
    print(f"Wrote {len(code_map)} CODE blocks -> {out}", file=sys.stderr)

    if args.py:
        py_out = out.with_suffix(".py")
        blocks = [to_python_stub(rid, steps) for rid, steps in code_map.items()]
        py_out.write_text("\n\n".join(blocks) + "\n")
        print(f"Wrote python stubs -> {py_out}", file=sys.stderr)


if __name__ == "__main__":
    main()