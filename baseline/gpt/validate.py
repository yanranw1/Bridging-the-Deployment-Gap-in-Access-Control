"""
validate.py  —  STEP 2 of 2

Reads generated_results.json (produced by generate.py) and evaluates each
generated CODE plan against its ground-truth CODE.

Per step it checks:
  - decision  (allow / deny)        exact
  - action    (the "api" field)     exact
  - resource  (required args only)  free-text fields {query, tone, instructions,
              subject, body, focus, description, message} matched by semantic
              similarity (>= SIM_THRESHOLD); all other required fields exact.
Plus the overall LOGIC (number of steps and their order).

Semantic matching uses OpenAI embeddings, so this step also needs an API key
UNLESS every generated plan is identical to ground truth (e.g. a dry-run echo),
in which case the offline difflib fallback is never exercised.

Usage:
    export OPENAI_API_KEY=sk-...
    python validate.py --input generated_results.json
    python validate.py --input generated_results.json --eval-file eval_results.json
"""

import argparse
import csv
import json
import os
import sys

import eval_common as ec
from eval_common import validate_record, empty_totals, accumulate, build_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="generated_results.json",
                    help="generated plans from generate.py")
    ap.add_argument("--embed-model", default="text-embedding-3-small",
                    help="OpenAI embedding model for semantic field matching")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="skip the API and use the offline difflib fallback "
                         "for semantic fields (stricter; for quick checks only)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override semantic match threshold (default %.2f)"
                         % ec.SIM_THRESHOLD)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--eval-file", default="eval_results.json")
    ap.add_argument("--eval-csv", default="eval_results.csv")
    args = ap.parse_args()

    if args.threshold is not None:
        ec.SIM_THRESHOLD = args.threshold

    with open(args.input) as f:
        payload = json.load(f)
    records = payload.get("records", payload)  # tolerate a bare list too

    # Decide whether we need an embeddings client.
    client = None
    needs_embeddings = not args.no_embeddings and any(
        r.get("generated_CODE") != r.get("ground_truth_CODE") for r in records
    )
    if needs_embeddings:
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                print("Note: OPENAI_API_KEY not set — falling back to offline "
                      "difflib similarity for free-text fields.", file=sys.stderr)
            else:
                client = OpenAI(api_key=key)
        except ImportError:
            print("Note: openai not installed — using offline difflib "
                  "similarity for free-text fields.", file=sys.stderr)

    results = []
    totals = empty_totals()

    for rec in records:
        gen = rec.get("generated_CODE", [])
        gt = rec.get("ground_truth_CODE", [])
        v = validate_record(client, args.embed_model, gen, gt)
        accumulate(totals, v)

        results.append({"id": rec["id"], "class": rec.get("class"),
                        "validation": v})

        flag = "EXACT" if v["record_exact_match"] else (
            "logic-mismatch" if not v["logic_ok"] else "partial")
        print(f"[{rec['id']}] {flag}  "
              f"steps {v['gen_steps']}/{v['gt_steps']}  "
              f"full-step {v['step_full_hits']}/{v['gt_steps']}")

    summary = build_summary(totals)

    os.makedirs(args.out_dir, exist_ok=True)
    eval_path = os.path.join(args.out_dir, args.eval_file)
    csv_path = os.path.join(args.out_dir, args.eval_csv)

    with open(eval_path, "w") as f:
        json.dump({"source": args.input,
                   "model": payload.get("model"),
                   "threshold": ec.SIM_THRESHOLD,
                   "summary": summary,
                   "results": results}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "gt_steps", "gen_steps", "logic_ok",
                    "step_full_hits", "decision_hits", "action_hits",
                    "resource_hits", "record_exact_match"])
        for r in results:
            v = r["validation"]
            w.writerow([r["id"], r["class"], v["gt_steps"], v["gen_steps"],
                        int(v["logic_ok"]), v["step_full_hits"],
                        v["field_hits"]["decision"], v["field_hits"]["action"],
                        v["field_hits"]["resource"],
                        int(v["record_exact_match"])])

    print("\n===== SUMMARY =====")
    for k, val in summary.items():
        print(f"{k:28s}: {val}")
    print(f"\nEval results -> {eval_path}")
    print(f"Eval CSV     -> {csv_path}")


if __name__ == "__main__":
    main()