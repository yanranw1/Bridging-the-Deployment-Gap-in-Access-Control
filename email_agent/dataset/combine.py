import json
import os
import re
from pathlib import Path

# ── Expected schema ────────────────────────────────────────────────────────────
REQUIRED_TOP_KEYS = {"id", "class", "NL", "ACP", "CODE"}

NL_REQUIRED_KEYS     = {"role", "text"}
ACP_REQUIRED_KEYS    = {"acp_id", "decision", "subject", "action",
                        "resource", "purpose", "condition", "policy_metadata"}
POLICY_REQUIRED_KEYS = {"contains_sensitive_information", "is_sensitive_action",
                        "has_conflicting_constraints", "contains_ambiguity"}
CODE_REQUIRED_KEYS   = {"acp_id", "api", "args", "decision", "reason"}

# ID format: <class_number>-<3-digit-sequence>  e.g. "4-001"
ID_PATTERN = re.compile(r"^(\d+)-(\d{3})$")

# ── Helpers ────────────────────────────────────────────────────────────────────

def validate_entry(entry: dict, entry_index: int) -> list[str]:
    """Return a list of schema violation messages for a single entry (empty = OK)."""
    issues = []

    missing_top = REQUIRED_TOP_KEYS - entry.keys()
    if missing_top:
        issues.append(f"  entry[{entry_index}] missing top-level keys: {missing_top}")
        return issues  # can't validate sub-fields if top keys missing

    # ── ID checks ──────────────────────────────────────────────────────────────
    entry_id  = entry.get("id", "")
    entry_cls = entry.get("class")
    match = ID_PATTERN.match(str(entry_id))

    # 1. Format check
    if not match:
        issues.append(
            f"  entry[{entry_index}] id '{entry_id}' does not match pattern '<class>-<###>'"
        )
    else:
        id_class_part = int(match.group(1))

        # 2. Class consistency check (id prefix must equal 'class' field)
        if id_class_part != entry_cls:
            issues.append(
                f"  entry[{entry_index}] id '{entry_id}' class prefix ({id_class_part}) "
                f"does not match 'class' field ({entry_cls})"
            )

    # ── NL ─────────────────────────────────────────────────────────────────────
    for i, nl in enumerate(entry.get("NL", [])):
        missing = NL_REQUIRED_KEYS - nl.keys()
        if missing:
            issues.append(f"  entry[{entry_index}].NL[{i}] missing keys: {missing}")

    # ── ACP ────────────────────────────────────────────────────────────────────
    for i, acp in enumerate(entry.get("ACP", [])):
        missing = ACP_REQUIRED_KEYS - acp.keys()
        if missing:
            issues.append(f"  entry[{entry_index}].ACP[{i}] missing keys: {missing}")
        pm = acp.get("policy_metadata", {})
        missing_pm = POLICY_REQUIRED_KEYS - pm.keys()
        if missing_pm:
            issues.append(
                f"  entry[{entry_index}].ACP[{i}].policy_metadata missing keys: {missing_pm}"
            )

    # ── CODE ───────────────────────────────────────────────────────────────────
    for i, code in enumerate(entry.get("CODE", [])):
        missing = CODE_REQUIRED_KEYS - code.keys()
        if missing:
            issues.append(f"  entry[{entry_index}].CODE[{i}] missing keys: {missing}")

    return issues


def validate_id_sequence(entries: list[dict], filename: str) -> list[str]:
    """
    Check that IDs within a file are sequential (e.g. 4-001, 4-002, 4-003…).
    Only validates entries whose IDs already passed the format check.
    """
    issues = []
    valid_ids = []

    for idx, entry in enumerate(entries):
        m = ID_PATTERN.match(str(entry.get("id", "")))
        if m:
            seq = int(m.group(2))
            valid_ids.append((idx, entry["id"], seq))

    for i, (idx, eid, seq) in enumerate(valid_ids):
        expected = i + 1          # sequences should start at 001 and increment by 1
        if seq != expected:
            issues.append(
                f"  entry[{idx}] id '{eid}' sequence {seq:03d} — "
                f"expected {expected:03d} (gap or out-of-order)"
            )
    return issues


def process_file(filepath: str, seen_ids: dict) -> tuple[list[dict], bool]:
    """
    Load a JSON file, print stats, validate schema + IDs.
    seen_ids  – shared dict {id_string: filename} for cross-file duplicate detection.
    Returns (entries, all_valid).
    """
    filename = Path(filepath).name
    print(f"\n{'─'*60}")
    print(f"  File : {filename}")

    if not os.path.exists(filepath):
        print(f"  ✗  File not found — skipping.")
        return [], False

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  ✗  Expected a JSON array at root — skipping.")
        return [], False

    count = len(data)
    print(f"  Entries : {count}")

    all_issues = []

    # Per-entry schema + id format/class checks
    for idx, entry in enumerate(data):
        all_issues.extend(validate_entry(entry, idx))

    # Sequence check (within this file)
    all_issues.extend(validate_id_sequence(data, filename))

    # Cross-file duplicate check
    for idx, entry in enumerate(data):
        eid = entry.get("id")
        if eid is not None:
            if eid in seen_ids:
                all_issues.append(
                    f"  entry[{idx}] id '{eid}' is a duplicate — "
                    f"first seen in {seen_ids[eid]}"
                )
            else:
                seen_ids[eid] = filename

    if all_issues:
        print(f"  Schema  : ✗  {len(all_issues)} issue(s) found")
        for issue in all_issues:
            print(issue)
        return data, False
    else:
        print(f"  Schema  : ✓  All {count} entries comply")
        return data, True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Configure paths ────────────────────────────────────────────────────────
    input_dir   = "."          # folder containing class1.json … class7.json
    output_file = "combined.json"
    num_files   = 7
    # ──────────────────────────────────────────────────────────────────────────

    file_paths = [
        os.path.join(input_dir, f"class{i}.json") for i in range(1, num_files + 1)
    ]

    combined    : list[dict] = []
    file_stats  : list[dict] = []
    seen_ids    : dict       = {}   # {id_string: filename} — shared across all files

    print("=" * 60)
    print("  JSON Combiner & Schema Validator")
    print("=" * 60)

    for path in file_paths:
        entries, valid = process_file(path, seen_ids)
        combined.extend(entries)
        file_stats.append({
            "file"   : Path(path).name,
            "entries": len(entries),
            "valid"  : valid,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'File':<20} {'Entries':>8}  {'Schema':>8}")
    print(f"  {'─'*18:<20} {'─'*6:>8}  {'─'*6:>8}")
    for s in file_stats:
        status = "✓ OK" if s["valid"] else "✗ issues"
        print(f"  {s['file']:<20} {s['entries']:>8}  {status:>8}")
    print(f"  {'─'*18:<20} {'─'*6:>8}")
    print(f"  {'TOTAL':<20} {len(combined):>8}")

    all_valid = all(s["valid"] for s in file_stats)
    print(f"\n  Overall schema compliance: {'✓ PASS' if all_valid else '✗ FAIL (see details above)'}")

    # ── Write output ───────────────────────────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n  Combined file written → {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()