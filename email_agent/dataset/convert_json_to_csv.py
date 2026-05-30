"""
convert_json_to_csv.py

Converts class1.json through class7.json to CSVs matching the format of collected.csv:
  columns: (index), input, acp, output

- input : all NL turns serialized, preserving role, text, and resource
- acp   : always 1
- output: ACP entries formatted as
          {decision: X; subject: X; action: X; resource: X; purpose: X; condition: X}
          multiple ACP entries joined by " | "

NL serialization format (no curly braces, no pipe):
  Single turn:  "Role: text"
  Agent turn with resource: "Role: text (key=value, key=value)"
  Multiple turns joined by " -> "

Place this script in the same folder as class1.json ... class7.json and run:
    python3 convert_json_to_csv.py

Output files:
  class1_converted.csv ... class7_converted.csv
  combined_converted.csv
  combined_train.csv  (80% stratified per source class)
  combined_test.csv   (20% stratified per source class)
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
    Serialize all NL turns in Llama 3 chat format:
      <|start_header_id|>role<|end_header_id|>\n\ntext<|eot_id|>

    Resource (if present) is appended inline to the text:
      text (key=value, key=value)
    """
    parts = []
    for turn in turns:
        role = turn.get("role", "").strip()
        text = turn.get("text", "").strip()
        resource = turn.get("resource")

        if resource and isinstance(resource, dict):
            kv = ", ".join(
                f"{k}={v}" for k, v in resource.items() if v is not None
            )
            if kv:
                text += f" ({kv})"

        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
        )

    return "".join(parts)


def format_acp(acp_entry: dict) -> str:
    """Format a single ACP dict. No curly braces in resource."""
    def fmt(val):
        return "none" if val is None else str(val).strip()

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
        resource="["+serialize_resource(acp_entry.get("resource"))+"]",
        purpose=fmt(acp_entry.get("purpose")),
        condition=fmt(acp_entry.get("condition")),
    )


def convert(input_path: str, output_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for record in data:
        nl_text   = serialize_nl(record.get("NL", []))
        acp_flag  = 1
        output_str = " | ".join(format_acp(a) for a in record.get("ACP", []))
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


def stratified_split(all_rows_with_class: list, test_ratio: float, seed: int):
    """
    Stratified split keeping equal per-source-class ratio.
    all_rows_with_class: list of (source_class_int, row_dict)
    acp in row_dict stays 1 throughout.
    """
    rng = random.Random(seed)

    by_class = {}
    for cls, row in all_rows_with_class:
        by_class.setdefault(cls, []).append(row)

    train, test = [], []
    for cls in sorted(by_class):
        cls_rows = by_class[cls][:]
        rng.shuffle(cls_rows)
        n_test = max(1, round(len(cls_rows) * test_ratio))
        test.extend(cls_rows[:n_test])
        train.extend(cls_rows[n_test:])
        print(f"  class {cls}: {len(cls_rows)} total → {len(cls_rows) - n_test} train, {n_test} test")

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aug_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "augmented_output"
    )

    combined_rows = []          # plain row dicts (acp=1)
    tagged_rows   = []          # (source_class, row_dict) for stratification

    # --- Convert each file individually ---
    for n in range(1, 8):
        input_path  = os.path.join(script_dir, f"class{n}.json")
        output_path = os.path.join(script_dir, f"class{n}_converted.csv")

        if not os.path.exists(input_path):
            print(f"[SKIP] class{n}.json not found.")
            continue

        print(f"[{n}/7] Converting class{n}.json ...")
        rows = convert(input_path, output_path)
        combined_rows.extend(rows)
        tagged_rows.extend((n, row) for row in rows)
        if n != 7:
            input_path  = os.path.join(aug_dir, f"augmented_class{n}.json")
            output_path = os.path.join(aug_dir, f"augmented_class{n}_converted.csv")
            if not os.path.exists(input_path):
                print(f"[SKIP] class{n}.json not found.")
                continue

            print(f"[{n}/6] Converting augmented class{n}.json ...")
            rows = convert(input_path, output_path)
            combined_rows.extend(rows)
            tagged_rows.extend((n, row) for row in rows)

    # --- Combined CSV (acp always 1) ---
    combined_path = os.path.join(script_dir, "combined_converted.csv")
    write_csv(combined_path, combined_rows)
    print(f"\nCombined: {len(combined_rows)} total rows → 'combined_converted.csv'")

    # --- Stratified train / test split ---
    print(f"\nSplitting with test_ratio={TEST_RATIO}, seed={RANDOM_SEED}:")
    train_rows, test_rows = stratified_split(tagged_rows, TEST_RATIO, RANDOM_SEED)

    write_csv(os.path.join(script_dir, "combined_train.csv"), train_rows)
    write_csv(os.path.join(script_dir, "combined_test.csv"),  test_rows)

    print(f"\nTrain: {len(train_rows)} rows → 'combined_train.csv'")
    print(f"Test:  {len(test_rows)} rows → 'combined_test.csv'")
    print("All done.")