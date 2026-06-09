"""
validate.py  —  STEP 2 of 2

Reads generated_results.json (produced by generate.py) and scores each
generated CODE plan against its ground-truth CODE.

Reports PER-RECORD accuracy (a record counts only if EVERY step matches):
  - decision_accuracy   all steps' decisions match
  - action_accuracy     all steps' actions match
  - resource_accuracy   all steps' resources match
  - all_match_accuracy   all steps fully match (decision AND action AND resource)

Usage:
    export OPENAI_API_KEY=sk-...
    python validate.py --input generated_results.json
"""

import argparse
import json
import os
import sys

import eval_common as ec
from eval_common import validate_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="generated_results.json")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="skip the API and use offline difflib similarity")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    if args.threshold is not None:
        ec.SIM_THRESHOLD = args.threshold

    with open(args.input) as f:
        payload = json.load(f)
    records = payload.get("records", payload)

    # Embeddings client only needed if some generated plan differs from gt.
    client = None
    needs_embeddings = not args.no_embeddings and any(
        r.get("generated_CODE") != r.get("ground_truth_CODE") for r in records
    )
    if needs_embeddings:
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                client = OpenAI(api_key=key)
            else:
                print("Note: OPENAI_API_KEY not set — using offline difflib "
                      "similarity for free-text fields.", file=sys.stderr)
        except ImportError:
            print("Note: openai not installed — using offline difflib "
                  "similarity for free-text fields.", file=sys.stderr)

    n = 0
    decision_ok = action_ok = resource_ok = all_ok = 0

    for rec in records:
        gen = rec.get("generated_CODE", [])
        gt = rec.get("ground_truth_CODE", [])
        v = validate_record(client, args.embed_model, gen, gt)
        steps = v["gt_steps"]

        n += 1
        d = v["field_hits"]["decision"] == steps
        a = v["field_hits"]["action"] == steps
        r = v["field_hits"]["resource"] == steps
        full = v["logic_ok"] and v["step_full_hits"] == steps

        decision_ok += int(d)
        action_ok += int(a)
        resource_ok += int(r)
        all_ok += int(full)

        print(f"[{rec['id']}] decision={int(d)} action={int(a)} "
              f"resource={int(r)} all={int(full)}")

    denom = n or 1
    print("\n===== SUMMARY (per record) =====")
    print(f"records             : {n}")
    print(f"decision_accuracy   : {decision_ok / denom:.4f}")
    print(f"action_accuracy     : {action_ok / denom:.4f}")
    print(f"resource_accuracy   : {resource_ok / denom:.4f}")
    print(f"all_match_accuracy  : {all_ok / denom:.4f}")


if __name__ == "__main__":
    main()