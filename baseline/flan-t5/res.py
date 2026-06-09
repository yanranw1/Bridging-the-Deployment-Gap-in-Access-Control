"""
Standalone summary printer.
Loads results.json (written by new_test.py) and prints the aggregate summary.

Usage:
    python print_summary.py [results.json]
"""

import json
import sys
from pathlib import Path


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



def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    results = json.loads(Path(path).read_text())
    print_summary(results)


if __name__ == "__main__":
    main()