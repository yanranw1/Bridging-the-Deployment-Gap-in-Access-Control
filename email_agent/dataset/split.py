"""
split_dataset.py
────────────────
Stratified 80 / 20 train-validation split of combined.json.
Each class contributes exactly 20% (rounded up) to the validation set,
so class distribution is preserved in both splits.

Usage:
    python split_dataset.py                        # defaults
    python split_dataset.py --val 0.2 --seed 42   # explicit options

Outputs:
    train.json   – 80% of each class
    val.json     – 20% of each class
"""

import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path: str, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratified_split(
    data      : list[dict],
    val_ratio : float = 0.20,
    seed      : int   = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split data into (train, val) stratified by the 'class' field.
    Uses math.ceil so every class always has at least 1 val entry.
    """
    random.seed(seed)

    # Group by class
    groups: dict = defaultdict(list)
    for entry in data:
        groups[entry.get("class")].append(entry)

    train_set, val_set = [], []

    for cls in sorted(groups.keys()):
        entries = groups[cls][:]
        random.shuffle(entries)                   # reproducible with seed

        n_val   = math.ceil(len(entries) * val_ratio)
        n_train = len(entries) - n_val

        val_set.extend(entries[:n_val])
        train_set.extend(entries[n_val:])

    return train_set, val_set


def print_report(
    groups    : dict,
    train_set : list[dict],
    val_set   : list[dict],
    val_ratio : float,
    seed      : int,
    train_path: str,
    val_path  : str,
) -> None:
    # Build per-class counts from the split results
    train_counts: dict = defaultdict(int)
    val_counts  : dict = defaultdict(int)
    for e in train_set:
        train_counts[e.get("class")] += 1
    for e in val_set:
        val_counts[e.get("class")] += 1

    print("\n" + "=" * 62)
    print("  Stratified Dataset Split")
    print(f"  Val ratio : {val_ratio*100:.0f}%   Seed : {seed}")
    print("=" * 62)
    print(f"  {'Class':<10} {'Total':>7} {'Train':>7} {'Val':>7}  {'Val%':>6}")
    print(f"  {'─'*8:<10} {'─'*5:>7} {'─'*5:>7} {'─'*5:>7}  {'─'*5:>6}")

    for cls in sorted(groups.keys()):
        total  = len(groups[cls])
        n_tr   = train_counts[cls]
        n_val  = val_counts[cls]
        pct    = n_val / total * 100
        print(f"  {str(cls):<10} {total:>7} {n_tr:>7} {n_val:>7}  {pct:>5.1f}%")

    total_all = len(train_set) + len(val_set)
    overall_pct = len(val_set) / total_all * 100
    print(f"  {'─'*8:<10} {'─'*5:>7} {'─'*5:>7} {'─'*5:>7}  {'─'*5:>6}")
    print(f"  {'TOTAL':<10} {total_all:>7} {len(train_set):>7} {len(val_set):>7}  {overall_pct:>5.1f}%")

    print(f"\n  ✓  train.json → {train_path}  ({len(train_set)} entries)")
    print(f"  ✓  val.json   → {val_path}  ({len(val_set)} entries)")
    print("=" * 62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val split")
    parser.add_argument("--input",  default="combined.json", help="Input combined JSON file")
    parser.add_argument("--train",  default="train.json",    help="Output train JSON file")
    parser.add_argument("--val",    default="val.json",      help="Output val JSON file")
    parser.add_argument("--ratio",  type=float, default=0.20, help="Validation ratio (default 0.20)")
    parser.add_argument("--seed",   type=int,   default=42,   help="Random seed (default 42)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"✗  Input file not found: {args.input}")
        return

    data = load(args.input)
    print(f"  Loaded {len(data)} entries from {args.input}")

    # Re-group for report
    groups: dict = defaultdict(list)
    for entry in data:
        groups[entry.get("class")].append(entry)

    train_set, val_set = stratified_split(data, val_ratio=args.ratio, seed=args.seed)

    save(args.train, train_set)
    save(args.val,   val_set)

    print_report(groups, train_set, val_set, args.ratio, args.seed, args.train, args.val)


if __name__ == "__main__":
    main()
