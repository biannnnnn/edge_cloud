#!/usr/bin/env python3
"""Collect the four fixed-ratio training and evaluation results into one CSV."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path("/home/liuheng/LLaMA-Factory")
OUTPUT = Path("/home/liuheng/edge_cloude/fixed_ratio_experiments/fixed_ratio_results.csv")


def find_latest_summary(percentage: int) -> Path | None:
    folder = ROOT / "fixed_ratio_eval" / f"fixed_ratio_{percentage:03d}" / "统计目录2"
    candidates = sorted(folder.glob("statistics_summary_local_*.txt"))
    return candidates[-1] if candidates else None


def extract(pattern: str, text: str, cast=float):
    match = re.search(pattern, text)
    return cast(match.group(1)) if match else None


def main() -> None:
    records = []
    for percentage in (20, 40, 60, 80):
        name = f"fixed_ratio_{percentage:03d}"
        train_results = ROOT / "saves" / f"qwen3-8b-{name}" / "lora" / "sft" / "train_results.json"
        train = json.loads(train_results.read_text()) if train_results.exists() else {}
        summary_path = find_latest_summary(percentage)
        summary = summary_path.read_text(encoding="utf-8") if summary_path else ""
        records.append(
            {
                "retention_ratio": percentage / 100,
                "train_loss": train.get("train_loss"),
                "train_runtime_sec": train.get("train_runtime"),
                "eval_games": extract(r"运行总局数:\s*(\d+)", summary, int),
                "success_rate_percent": extract(r"成功率:\s*\d+/\d+\s*\(([\d.]+)%\)", summary),
                "avg_steps": extract(r"平均步数:\s*([\d.]+)", summary),
                "avg_output_tokens": extract(r"平均消耗Tokens:\s*([\d.]+)", summary),
                "avg_duration_sec": extract(r"平均每局耗时:\s*([\d.]+)", summary),
                "summary_path": str(summary_path) if summary_path else None,
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(OUTPUT)


if __name__ == "__main__":
    main()
