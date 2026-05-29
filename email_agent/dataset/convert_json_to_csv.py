"""
convert_json_to_csv.py

Converts class1.json through class7.json to CSVs matching the format of collected.csv:
  columns: (index), input, acp, output

- input : the NL user text
- acp   : the class field (1 or 0)
- output: ACP entries formatted as
          {decision: X; subject: X; action: X; resource: X; purpose: X; condition: X}
          multiple ACPs joined by " | "

Place this script in the same folder as class1.json ... class7.json and run:
    python3 convert_json_to_csv.py

Output files:
  class1_converted.csv ... class7_converted.csv
  combined_converted.csv
  combined_train.csv  (80% stratified per class)
  combined_test.csv   (20% stratified per class)
"""

import json
import csv
import os
import random

RANDOM_SEED = 42
TEST_RATIO = 0.2


def format_acp(acp_entry: dict) -> str:
    """Format a single ACP dict into the target string representation."""
    def fmt(val):
        if val is None:
            return "none"
        return str(val).strip()

    return (
        "{{decision: {decision}; "
        "subject: {subject}; "
        "action: {action}; "
        "resource: {resource}; "
        "purpose: {purpose}; "
        "condition: {condition}}}"
    ).format(
        decision=fmt(acp_entry.get("decision")),
        subject=fmt(acp_entry.get("subject")),
        action=fmt(acp_entry.get("action")),
        resource=fmt(acp_entry.get("resource")),
        purpose=fmt(acp_entry.get("purpose")),
        condition=fmt(acp_entry.get("condition")),
    )


def convert(input_path: str, output_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for record in data:
        nl_text = " ".join(
            turn["text"] for turn in record.get("NL", []) if turn.get("text")
        )
        acp_flag = record.get("class", "")
        acp_entries = record.get("ACP", [])
        output_str = " | ".join(format_acp(a) for a in acp_entries)
        rows.append({"input": nl_text, "acp": acp_flag, "output": output_str})

    write_csv(output_path, rows)
    print(f"  {len(rows)} rows written to '{os.path.basename(output_path)}'.")
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "input", "acp", "output"])
        for i, row in enumerate(rows):
            writer.writerow([i, row["input"], row["acp"], row["output"]])


def stratified_split(rows: list, test_ratio: float, seed: int):
    """
    Split rows into train/test with equal per-class representation.
    Each class contributes exactly ceil/floor(test_ratio * class_size) rows to test.
    """
    rng = random.Random(seed)

    # Group rows by their source class (acp field = class number 1-7)
    by_class = {}
    for row in rows:
        key = row["acp"]
        by_class.setdefault(key, []).append(row)

    train, test = [], []
    for cls, cls_rows in sorted(by_class.items()):
        shuffled = cls_rows[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
        print(f"  class {cls}: {len(cls_rows)} total → {len(shuffled) - n_test} train, {n_test} test")

    # Shuffle final sets so classes are interleaved
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combined_rows = []

    # --- Convert each file individually ---
    for n in range(1, 8):
        input_path = os.path.join(script_dir, f"class{n}.json")
        output_path = os.path.join(script_dir, f"class{n}_converted.csv")

        if not os.path.exists(input_path):
            print(f"[SKIP] class{n}.json not found.")
            continue

        print(f"[{n}/7] Converting class{n}.json ...")
        rows = convert(input_path, output_path)
        # Tag each row with its source class for stratification
        for row in rows:
            row["acp"] = n  # overwrite with integer class label
        combined_rows.extend(rows)

    # --- Write combined CSV ---
    combined_path = os.path.join(script_dir, "combined_converted.csv")
    write_csv(combined_path, combined_rows)
    print(f"\nCombined: {len(combined_rows)} total rows → 'combined_converted.csv'")

    # --- Stratified train / test split ---
    print(f"\nSplitting with test_ratio={TEST_RATIO}, seed={RANDOM_SEED}:")
    train_rows, test_rows = stratified_split(combined_rows, TEST_RATIO, RANDOM_SEED)

    train_path = os.path.join(script_dir, "combined_train.csv")
    test_path  = os.path.join(script_dir, "combined_test.csv")
    write_csv(train_path, train_rows)
    write_csv(test_path,  test_rows)

    print(f"\nTrain: {len(train_rows)} rows → 'combined_train.csv'")
    print(f"Test:  {len(test_rows)} rows → 'combined_test.csv'")
    print("All done.")