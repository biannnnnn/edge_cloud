#!/usr/bin/env python3
"""Create a balanced immutable snapshot from in-progress fixed-ratio JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # The producer may be appending the final line at this instant.
                continue
            index = int(row["source_index"])
            if row.get("exact_length") is not True:
                continue
            if int(row["actual_tokens"]) != int(row["target_tokens"]):
                continue
            if float(row.get("model_coverage", 0.0)) < 0.95:
                continue
            rows[index] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"snapshot output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows_by_ratio = {
        percentage: load(source / f"fixed_ratio_{percentage:03d}.jsonl")
        for percentage in (20, 40, 60, 80)
    }
    common = sorted(set.intersection(*(set(rows) for rows in rows_by_ratio.values())))
    if not common:
        raise RuntimeError("no common validated samples")

    files = {}
    for percentage, rows in rows_by_ratio.items():
        path = output / f"fixed_ratio_{percentage:03d}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for index in common:
                handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
        files[path.name] = {
            "records": len(common),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source),
        "common_records": len(common),
        "source_indexes": common,
        "files": files,
    }
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "common_records": len(common)}))


if __name__ == "__main__":
    main()
