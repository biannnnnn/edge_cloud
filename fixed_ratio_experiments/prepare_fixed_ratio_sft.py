#!/usr/bin/env python3
"""Convert fixed-ratio JSONL files to isolated LLaMA-Factory datasets."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT_DIR = "/home/liuheng/edge_cloude/fixed_ratio_experiments/data"
DEFAULT_OUTPUT_DIR = "/home/liuheng/LLaMA-Factory/data/fixed_ratio"
DEFAULT_PROMPT_SOURCE = "/home/liuheng/edge_cloude/game2_refine.py"
DEFAULT_REFERENCE = "/home/liuheng/LLaMA-Factory/data/game2_dataset1_f.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt_source", default=DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--reference_dataset", default=DEFAULT_REFERENCE)
    parser.add_argument("--ratios", nargs="+", type=int, default=[20, 40, 60, 80])
    return parser.parse_args()


def load_prompt_builder(source_path: str):
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=source_path)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_prompt_from_state"
    ]
    if len(selected) != 1:
        raise RuntimeError("build_prompt_from_state was not found uniquely")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
    }
    exec(compile(module, source_path, "exec"), namespace)
    return namespace["build_prompt_from_state"]


def load_system_prompt(reference_path: str) -> str:
    rows = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    for message in rows[0].get("messages", []):
        if message.get("role") == "system":
            return str(message.get("content", ""))
    raise RuntimeError("reference dataset has no system prompt")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["source_index"]))
    return rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_prompt = load_prompt_builder(args.prompt_source)
    system_prompt = load_system_prompt(args.reference_dataset)
    dataset_info: dict[str, Any] = {}

    source_by_percentage = {
        percentage: load_jsonl(input_dir / f"fixed_ratio_{percentage:03d}.jsonl")
        for percentage in args.ratios
    }
    common_indexes = set.intersection(
        *(
            {int(row["source_index"]) for row in rows}
            for rows in source_by_percentage.values()
        )
    )
    if not common_indexes:
        raise RuntimeError("the fixed-ratio datasets have no common source indexes")
    print(f"balanced common subset: {len(common_indexes)} records")

    for percentage in args.ratios:
        name = f"fixed_ratio_{percentage:03d}"
        source_rows = [
            row
            for row in source_by_percentage[percentage]
            if int(row["source_index"]) in common_indexes
        ]
        converted = []
        for row in source_rows:
            if row.get("exact_length") is not True:
                raise ValueError(f"{name} source_index={row.get('source_index')} is not exact")
            if int(row["actual_tokens"]) != int(row["target_tokens"]):
                raise ValueError(f"{name} source_index={row.get('source_index')} token mismatch")
            user_prompt = build_prompt(
                row["state"],
                row["vip_next_pos"],
                formation_limit=3,
                grid_size=20,
                max_steps=100,
            )
            decision = str(row["decision"]).strip()
            assistant = f"{row['fixed_cot']}\n\n```{decision}```"
            converted.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant},
                    ]
                }
            )

        output_path = output_dir / f"{name}.json"
        output_path.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        dataset_info[name] = {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        print(f"{name}: {len(converted)} records -> {output_path}")

    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
