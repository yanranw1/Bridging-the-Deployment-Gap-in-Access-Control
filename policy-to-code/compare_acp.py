"""
ACP CSV → JSON conversion + exact-match check against gold JSON.

Reads:
    combined_test.csv   (column 'output' = flat ACP text, '|'-separated)
    combined_test.json  (list of records, each with an 'ACP' gold list)

For each row it converts the CSV 'output' text with acp_to_json.convert()
and compares to the gold ACP list at the same index. policy_metadata is
ignored on the gold side, since it cannot be derived from the flat text.
"""

import argparse
import csv
import json
import sys

from acp_to_json import convert


def strip_meta(acp_list):
    """Return a copy of an ACP list with policy_metadata removed from each obj."""
    return [{k: v for k, v in a.items() if k != "policy_metadata"} for a in acp_list]


def normalize(acp_list):
    """Canonical form for comparison: JSON with sorted keys."""
    return json.dumps(acp_list, ensure_ascii=False, sort_keys=True)


def load_csv_outputs(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [row.get("output", "") or "" for row in csv.DictReader(f)]


def load_gold(path):
    data = json.load(open(path, encoding="utf-8"))
    return [rec.get("ACP", []) for rec in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_train.csv")
    ap.add_argument("--json", default="/home/ubuntu/agentv-main/email_agent/dataset/combined_train.json")
    ap.add_argument("--output", default="comparison_results.json")
    ap.add_argument("--verbose", action="store_true",
                    help="Print a diff line for every mismatch.")
    args = ap.parse_args()

    csv_outputs = load_csv_outputs(args.csv)
    gold_acps   = load_gold(args.json)

    n = min(len(csv_outputs), len(gold_acps))
    if len(csv_outputs) != len(gold_acps):
        print(f"WARNING: row count mismatch — CSV={len(csv_outputs)} "
              f"JSON={len(gold_acps)}; comparing first {n}.", file=sys.stderr)

    results = []
    matched = 0

    for i in range(n):
        converted = convert(csv_outputs[i])
        gold      = strip_meta(gold_acps[i])

        is_match = normalize(converted) == normalize(gold)
        matched += is_match

        results.append({
            "index": i,
            "match": is_match,
            "converted": converted,
            "gold": gold,
        })

        if args.verbose and not is_match:
            print(f"\n[{i}] MISMATCH")
            print("  CONVERTED:", normalize(converted))
            print("  GOLD     :", normalize(gold))

    print(f"\n{'='*60}")
    print(f"RESULTS ({n} rows)")
    print(f"  Exact match: {matched:3d}  ({100*matched/n:.1f}%)")
    print(f"{'='*60}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results → {args.output}")


if __name__ == "__main__":
    main()