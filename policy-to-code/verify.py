#!/usr/bin/env python3
"""Derive CODE from each ACP block and measure exact-match rate vs ground-truth CODE.

Match is computed on the executable core of each step: (api, args, decision).
- Step-level: fraction of individual steps that match.
- Record-level: fraction of records whose full ordered step sequence matches.

Usage: python verify.py  /home/ubuntu/agentv-main/email_agent/dataset/combined_test.json
"""

import argparse
import ast
import json
import re
from pathlib import Path


def parse_resource(resource):
    """Parse an ACP `resource` string of `key: value, key: value` into a dict.

    Values are coerced to int/float/bool when they look numeric/boolean,
    matching how the ground-truth CODE.args are typed.
    """
    if not isinstance(resource, str):
        return resource if isinstance(resource, dict) else {}
    args = {}
    # split on commas that separate top-level key: value pairs
    parts = re.split(r",\s*(?=[A-Za-z_][\w]*\s*:)", resource)
    for part in parts:
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        args[key] = coerce(val)
    return args


def coerce(val):
    low = val.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def derive_code(acp_steps):
    """Build CODE-style steps from ACP steps."""
    out = []
    for s in acp_steps:
        out.append({
            "acp_id": s.get("acp_id"),
            "api": s.get("action"),
            "args": parse_resource(s.get("resource", "")),
            "decision": s.get("decision"),
        })
    return out


def core(step):
    """Comparable core of a step, normalized for type/whitespace."""
    args = step.get("args", {})
    norm = {str(k).strip(): normalize(v) for k, v in args.items()}
    return (step.get("api"), tuple(sorted(norm.items())), step.get("decision"))


def normalize(v):
    if isinstance(v, str):
        return coerce(v.strip())
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--show-mismatch", action="store_true")
    args = ap.parse_args()

    records = json.loads(args.input.read_text())
    if isinstance(records, dict):
        records = [records]

    total_steps = matched_steps = 0
    total_recs = matched_recs = 0
    rec_with_code = 0
    mismatches = []

    for r in records:
        gt = r.get("CODE", []) or []
        if gt:
            rec_with_code += 1
        derived = derive_code(r.get("ACP", []) or [])

        gt_core = [core(s) for s in gt]
        dv_core = [core(s) for s in derived]

        # step-level (positional)
        n = max(len(gt_core), len(dv_core))
        total_steps += n
        step_ok = 0
        for i in range(n):
            if i < len(gt_core) and i < len(dv_core) and gt_core[i] == dv_core[i]:
                step_ok += 1
        matched_steps += step_ok

        # record-level (full sequence identical)
        total_recs += 1
        if gt_core == dv_core:
            matched_recs += 1
        elif args.show_mismatch:
            mismatches.append((r.get("id"), gt, derived))

    print(f"Records total            : {total_recs}")
    print(f"  of which have CODE     : {rec_with_code}")
    print(f"Step-level exact match   : {matched_steps}/{total_steps} "
          f"= {matched_steps/total_steps:.1%}" if total_steps else "n/a")
    print(f"Record-level exact match : {matched_recs}/{total_recs} "
          f"= {matched_recs/total_recs:.1%}" if total_recs else "n/a")

    if args.show_mismatch and mismatches:
        print("\n--- Mismatched records ---")
        for rid, gt, dv in mismatches[:10]:
            print(f"\n[{rid}]")
            for g, d in zip(gt, dv):
                if core(g) != core(d):
                    print("  GT :", g.get("api"), g.get("args"), g.get("decision"))
                    print("  DRV:", d.get("api"), d.get("args"), d.get("decision"))


if __name__ == "__main__":
    main()