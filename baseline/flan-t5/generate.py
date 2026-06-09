"""
generate.py  —  STEP 1 of 2

Reads the test dataset (combined_test.json) and, for every record, asks the
OpenAI API to generate a CODE plan from the natural-language conversation
(the "NL" field). Writes generated_results.json, which validate.py then scores.

Each output record carries:
  - id, class
  - NL                (the input conversation)
  - generated_CODE    (the model's plan)
  - ground_truth_CODE (the dataset "CODE" field, copied through for scoring)

Generation logic (prompt, catalogue, parsing) lives in eval_common, so this
file and validate.py stay in sync.

Usage:
    export OPENAI_API_KEY=sk-...
    python generate.py --model gpt-4o
    python generate.py --model gpt-4o --limit 20 --out-dir runs/

    # smoke test without spending tokens (echoes ground truth back):
    python generate.py --dry-run
"""

import argparse
import json
import os
import sys

import eval_common as ec
from eval_common import generate_code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/home/ubuntu/agentv-main/email_agent/dataset/combined_with0_test.json",
        help="test dataset with NL (input) and CODE (ground truth) per record",
    )
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI chat model used to generate plans")
    ap.add_argument("--out-dir", default=".",
                    help="directory for output files")
    ap.add_argument("--generated-file", default="generated_results.json",
                    help="where to save the model-generated CODE plans")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N records")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the API; echo ground truth back (tests the harness)")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    # client = None
    # if not args.dry_run:
    #     try:
    #         from openai import OpenAI
    #     except ImportError:
    #         sys.exit("openai package not installed. Run: pip install openai")
    #     key = os.environ.get("OPENAI_API_KEY")
    #     if not key:
    #         sys.exit("Set OPENAI_API_KEY (or use --dry-run).")
    #     client = OpenAI(api_key=key)

    generated_records = []

    for rec in data:
        nl = rec.get("NL", [])
        gt = rec.get("CODE", [])

        # if args.dry_run:
        #     generated = json.loads(json.dumps(gt))  # echo (perfect score)
        # else:
        generated = generate_code(nl)
        print("generated",generated)
        

        generated_records.append({
            "id": rec["id"],
            "class": rec.get("class"),
            "NL": nl,
            "generated_CODE": generated,
            "ground_truth_CODE": gt,
        })

        print(f"[{rec['id']}] generated {len(generated)} step(s) "
              f"(gt {len(gt)})")

    os.makedirs(args.out_dir, exist_ok=True)
    gen_path = os.path.join(args.out_dir, args.generated_file)

    with open(gen_path, "w") as f:
        json.dump({"model": "dry-run" if args.dry_run else args.model,
                   "records": generated_records}, f, indent=2)

    print(f"\nGenerated plans -> {gen_path}")
    print(f"Records         -> {len(generated_records)}")
    print("Next: python validate.py --input %s" % gen_path)


if __name__ == "__main__":
    main()