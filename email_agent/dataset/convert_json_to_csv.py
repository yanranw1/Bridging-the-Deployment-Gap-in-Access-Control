"""
convert_json_to_csv.py

Converts class1.json through class7.json to CSVs matching the format of collected.csv:
  columns: (index), input, acp, output

Outputs:
  class1_converted.csv ... class7_converted.csv
  combined_converted.csv      / combined.json
  combined_train.csv          / combined_train.json
  combined_test.csv           / combined_test.json
"""

import json
import csv
import os
import random

RANDOM_SEED = 42
TEST_RATIO = 0.1

def serialize_resource(val) -> str:
    """
    Serialize a resource value cleanly — no curly braces.
      - None  → "none"
      - str   → as-is
      - dict  → "key=value, key=value"  (None values omitted)
    """
    if val is None:
        return "none"
    if isinstance(val, str):
        # normalize "key: value" → "key=value"
        if ": " in val:
            k, v = val.split(": ", 1)
            return f"{k}={v}"
        return val.strip()
    if isinstance(val, dict):
        parts = [f"{k}={v}" for k, v in val.items() if v is not None]
        return ", ".join(parts) if parts else "none"
    return str(val).strip()


def serialize_nl(turns: list) -> str:
    """
    Serialize all NL turns preserving role, text, and resource.
    No curly braces, no pipe characters used.

    Per turn format:
      - No resource : "Role: text"
      - With resource: "Role: text (key=value, key=value)"

    Multiple turns are joined with " -> " to show conversation flow.
    """
    parts = []
    for turn in turns:
        role = turn.get("role", "").strip()
        text = turn.get("text", "").strip()
        resource = turn.get("resource")

        turn_str = f"{role}: {text}"

        if resource and isinstance(resource, dict):
            kv = ", ".join(
                f"{k}={v}" for k, v in resource.items() if v is not None
            )
            if kv:
                turn_str += f" ({kv})"

        parts.append(turn_str)

    return "\n".join(parts)



def format_acp(acp_entry: dict) -> str:
    """Format a single ACP entry — no surrounding braces (added at join time)."""
    def fmt(val):
        return "none" if val is None else str(val).strip()

    return (
        "decision: {decision}; "
        "subject: {subject}; "
        "action: {action}; "
        "resource: {resource}; "
        "purpose: {purpose}; "
        "condition: {condition}"
    ).format(
        decision=fmt(acp_entry.get("decision")),
        subject=fmt(acp_entry.get("subject")),
        action=fmt(acp_entry.get("action")),
        resource=serialize_resource(acp_entry.get("resource")),
        purpose=fmt(acp_entry.get("purpose")),
        condition=fmt(acp_entry.get("condition")),
    )



def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "input", "acp", "output"])
        for i, row in enumerate(rows):
            writer.writerow([i, row["input"], row["acp"], row["output"]])


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratified_split(tagged, test_ratio: float, seed: int):
    """
    Stratified split keeping equal per-source-class ratio.
    tagged: list of (source_class_int, row_dict, json_record)
    Returns (train_rows, test_rows, train_records, test_records),
    where the CSV rows and JSON records stay aligned per split.
    """
    rng = random.Random(seed)

    by_class = {}
    for cls, row, rec in tagged:
        by_class.setdefault(cls, []).append((row, rec))

    train, test = [], []
    for cls in sorted(by_class):
        items = by_class[cls][:]
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_ratio))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
        print(f"  class {cls}: {len(items)} total -> "
              f"{len(items) - n_test} train, {n_test} test")

    rng.shuffle(train)
    rng.shuffle(test)

    train_rows = [r for r, _ in train]
    test_rows = [r for r, _ in test]
    train_records = [rec for _, rec in train]
    test_records = [rec for _, rec in test]
    return train_rows, test_rows, train_records, test_records


def convert(input_path: str, output_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for record in data:
        nl_text    = serialize_nl(record.get("NL", []))
        acp_flag   = 1
        joined     = " | ".join(format_acp(a) for a in record.get("ACP", []))
        output_str = "{" + joined + "}"                  # ← single braces around all
        rows.append({"input": nl_text, "acp": acp_flag, "output": output_str})

    write_csv(output_path, rows)
    print(f"  {len(rows)} rows written to '{os.path.basename(output_path)}'.")
    return rows


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    combined_rows = []
    combined_json_records = []
    tagged_rows = []  # (source_class, row_dict, json_record)

    # --- Convert each file individually ---
    for n in range(1, 8):
        input_path = os.path.join(script_dir, f"class{n}.json")
        output_path = os.path.join(script_dir, f"class{n}_converted.csv")

        if not os.path.exists(input_path):
            print(f"[SKIP] class{n}.json not found.")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        print(f"[{n}/7] Converting class{n}.json ...")
        rows = convert(input_path, output_path)

        combined_rows.extend(rows)
        combined_json_records.extend(original_data)
        tagged_rows.extend(
            (n, row, rec) for row, rec in zip(rows, original_data)
        )

    # --- Combined CSV + JSON ---
    write_csv(os.path.join(script_dir, "combined_converted.csv"), combined_rows)
    write_json(os.path.join(script_dir, "combined.json"), combined_json_records)
    print(f"\nCombined: {len(combined_rows)} rows -> "
          f"'combined_converted.csv' / 'combined.json'")

    # --- Stratified train / test split (CSV + JSON) ---
    print(f"\nSplitting with test_ratio={TEST_RATIO}, seed={RANDOM_SEED}:")
    train_rows, test_rows, train_recs, test_recs = stratified_split(
        tagged_rows, TEST_RATIO, RANDOM_SEED
    )

    write_csv(os.path.join(script_dir, "combined_train.csv"), train_rows)
    write_csv(os.path.join(script_dir, "combined_test.csv"), test_rows)
    write_json(os.path.join(script_dir, "combined_train.json"), train_recs)
    write_json(os.path.join(script_dir, "combined_test.json"), test_recs)

    print(f"\nTrain: {len(train_rows)} rows -> "
          f"'combined_train.csv' / 'combined_train.json'")
    print(f"Test:  {len(test_rows)} rows -> "
          f"'combined_test.csv' / 'combined_test.json'")
    print("All done.")