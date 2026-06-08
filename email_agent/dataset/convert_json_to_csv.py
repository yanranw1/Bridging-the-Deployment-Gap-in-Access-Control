"""
convert_json_to_csv.py

Converts classN.json to CSVs matching the format of collected.csv:
  columns: (index), input, acp, output

Produces TWO versions of the combined train/val/test splits:
  1. WITHOUT class0  -> files prefixed "combined_no0_*"
  2. WITH    class0  -> files prefixed "combined_with0_*"

Because the split is stratified with a PER-CLASS deterministic RNG
(seed + class_id), classes 1-7 receive EXACTLY the same rows in
train/val/test in both versions. The only difference between the two
versions is that the "with0" version additionally contains class0's rows.

Outputs:
  class0_converted.csv ... class7_converted.csv
  combined_no0_converted.csv    / combined_no0.json
  combined_no0_train.csv        / combined_no0_train.json
  combined_no0_val.csv          / combined_no0_val.json
  combined_no0_test.csv         / combined_no0_test.json
  combined_with0_converted.csv  / combined_with0.json
  combined_with0_train.csv      / combined_with0_train.json
  combined_with0_val.csv        / combined_with0_val.json
  combined_with0_test.csv       / combined_with0_test.json
"""

import json
import csv
import os
import random

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def serialize_resource(val) -> str:
    """
    Serialize a resource value cleanly - no curly braces.
      - None  -> "none"
      - str   -> converts all "key: value" or "key=value" fragments consistently
      - dict  -> "key=value, key=value"  (None values omitted)
    """
    if val is None:
        return "none"
    
    if isinstance(val, dict):
        parts = [f"{k}={v}" for k, v in val.items() if v is not None]
        return ", ".join(parts) if parts else "none"
        
    if isinstance(val, str):
        val = val.strip()
        # If it looks like JSON/Dict format but passed as a string, clean it up
        if val.startswith("{") and val.endswith("}"):
            val = val[1:-1].strip()
            
        # Split by comma to catch individual key-value pairs
        pairs = [p.strip() for p in val.split(",") if p.strip()]
        parts = []
        for pair in pairs:
            if ": " in pair:
                k, v = pair.split(": ", 1)
                parts.append(f"{k.strip()}={v.strip()}")
            elif "=" in pair:
                k, v = pair.split("=", 1)
                parts.append(f"{k.strip()}={v.strip()}")
            else:
                parts.append(pair)
        return ", ".join(parts) if parts else "none"
        
    return str(val).strip()


def serialize_nl(turns: list) -> str:
    """
    Serialize all NL turns preserving role, text, and resource.
      - No resource : "Role: text"
      - With resource: "Role: text (key=value, key=value)"
    Multiple turns joined with newline.
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
    """Format a single ACP entry - no surrounding braces (added at join time)."""
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


def split_one_class(items, train_ratio, val_ratio, test_ratio, seed, cls):
    """
    Split a single class's items deterministically using a per-class RNG.
    Returns (train, val, test) lists of (row, rec) tuples.

    Because the RNG depends only on (seed, cls) and the input items, a class
    always gets the same partition regardless of which other classes exist.
    """
    rng = random.Random(seed + cls)
    items = items[:]
    rng.shuffle(items)

    n = len(items)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)

    if n >= 10:
        n_test = max(1, n_test)
        n_val = max(1, n_val)

    n_train = n - n_val - n_test

    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]

    print(
        f"  class {cls}: {n} total -> "
        f"{n_train} train, {n_val} val, {n_test} test"
    )
    return train, val, test


def stratified_split(tagged, classes, train_ratio, val_ratio,
                     test_ratio, seed):
    """
    Stratified split over the given `classes` only.

    tagged: list of (source_class_int, row_dict, json_record)
    classes: iterable of class ids to INCLUDE in this version.

    Each included class is split independently via split_one_class, so its
    partition is identical no matter which other classes are included.

    Returns:
        train_rows, val_rows, test_rows,
        train_records, val_records, test_records
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    by_class = {}
    for cls, row, rec in tagged:
        by_class.setdefault(cls, []).append((row, rec))

    train, val, test = [], [], []

    for cls in sorted(c for c in classes if c in by_class):
        t, v, te = split_one_class(
            by_class[cls], train_ratio, val_ratio, test_ratio, seed, cls
        )
        train.extend(t)
        val.extend(v)
        test.extend(te)

    # Final merge-shuffle on base seed for reproducible combined ordering.
    final_rng = random.Random(seed)
    final_rng.shuffle(train)
    final_rng.shuffle(val)
    final_rng.shuffle(test)

    return (
        [r for r, _ in train],
        [r for r, _ in val],
        [r for r, _ in test],
        [rec for _, rec in train],
        [rec for _, rec in val],
        [rec for _, rec in test],
    )


def convert(input_path: str, output_path: str) -> list:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for record in data:
        nl_text = serialize_nl(record.get("NL", []))
        acp_flag = 0 if len(record.get("ACP", [])) == 0 else 1
        joined = " | ".join(format_acp(a) for a in record.get("ACP", []))
        output_str = "{" + joined + "}"
        rows.append({"input": nl_text, "acp": acp_flag, "output": output_str})

    write_csv(output_path, rows)
    print(f"  {len(rows)} rows written to '{os.path.basename(output_path)}'.")
    return rows


def write_version(script_dir, prefix, tagged, classes):
    """Run a stratified split over `classes` and write all output files."""
    print(
        f"\n=== Version '{prefix}' (classes: {sorted(classes)}) | "
        f"train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}, "
        f"seed={RANDOM_SEED} ==="
    )

    # combined (all included rows, in class order)
    combined_rows = [row for cls, row, _ in tagged if cls in classes]
    combined_recs = [rec for cls, _, rec in tagged if cls in classes]
    write_csv(os.path.join(script_dir, f"combined_{prefix}_converted.csv"),
              combined_rows)
    write_json(os.path.join(script_dir, f"combined_{prefix}.json"),
               combined_recs)

    (train_rows, val_rows, test_rows,
     train_recs, val_recs, test_recs) = stratified_split(
        tagged, classes, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    write_csv(os.path.join(script_dir, f"combined_{prefix}_train.csv"), train_rows)
    write_csv(os.path.join(script_dir, f"combined_{prefix}_val.csv"), val_rows)
    write_csv(os.path.join(script_dir, f"combined_{prefix}_test.csv"), test_rows)
    write_json(os.path.join(script_dir, f"combined_{prefix}_train.json"), train_recs)
    write_json(os.path.join(script_dir, f"combined_{prefix}_val.json"), val_recs)
    write_json(os.path.join(script_dir, f"combined_{prefix}_test.json"), test_recs)

    print(
        f"  -> {len(train_rows)} train / {len(val_rows)} val / "
        f"{len(test_rows)} test"
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    tagged_rows = []  # (source_class, row_dict, json_record)

    # --- Convert each file individually (class0 .. class7) ---
    for n in range(0, 8):
        input_path = os.path.join(script_dir, f"class{n}.json")
        output_path = os.path.join(script_dir, f"class{n}_converted.csv")

        if not os.path.exists(input_path):
            print(f"[SKIP] class{n}.json not found.")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        print(f"[class {n}] Converting class{n}.json ...")
        rows = convert(input_path, output_path)
        tagged_rows.extend(
            (n, row, rec) for row, rec in zip(rows, original_data)
        )

    present_classes = sorted({cls for cls, _, _ in tagged_rows})

    # --- Version 1: WITHOUT class0 (classes 1-7) ---
    write_version(
        script_dir, "no0", tagged_rows,
        classes=[c for c in present_classes if c != 0],
    )

    # --- Version 2: WITH class0 (classes 0-7) ---
    # Classes 1-7 get IDENTICAL train/val/test rows as version 1, because
    # each class's split is computed independently from (seed + cls).
    write_version(
        script_dir, "with0", tagged_rows,
        classes=present_classes,
    )

    print("\nAll done. Classes 1-7 have identical splits across both versions.")