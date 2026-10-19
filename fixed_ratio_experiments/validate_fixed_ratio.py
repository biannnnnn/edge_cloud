#!/usr/bin/env python3
"""Audit exact token ratios without printing private sample text."""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--tokenizer", default="/data3/liuyuan/models/Qwen3-8B")
    parser.add_argument("--expected_records", type=int, default=792)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    failed = False
    for percentage in (20, 40, 60, 80):
        path = Path(args.data_dir) / f"fixed_ratio_{percentage:03d}.jsonl"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        errors = []
        deviations = []
        indexes = set()
        for row in rows:
            index = int(row["source_index"])
            if index in indexes:
                errors.append(f"duplicate source_index={index}")
            indexes.add(index)
            original = len(tokenizer.encode(row["original_cot"], add_special_tokens=False))
            actual = len(tokenizer.encode(row["fixed_cot"], add_special_tokens=False))
            target = max(1, round(original * percentage / 100))
            deviations.append(actual / original - percentage / 100)
            if actual != target or actual != int(row["actual_tokens"]):
                errors.append(f"source_index={index}: original={original} target={target} actual={actual}")
        if len(rows) != args.expected_records:
            errors.append(f"records={len(rows)}, expected={args.expected_records}")
        print(
            json.dumps(
                {
                    "ratio": percentage / 100,
                    "records": len(rows),
                    "unique_source_indexes": len(indexes),
                    "exact": not errors,
                    "max_abs_ratio_rounding_error": max(map(abs, deviations), default=0.0),
                    "median_actual_ratio": statistics.median(
                        [percentage / 100 + value for value in deviations]
                    ) if deviations else None,
                    "errors": errors[:10],
                },
                ensure_ascii=False,
            )
        )
        failed = failed or bool(errors)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
