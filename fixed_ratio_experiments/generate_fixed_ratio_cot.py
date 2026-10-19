#!/usr/bin/env python3
"""Generate CoT datasets with an exact student-token retention ratio."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer


DEFAULT_INPUT = "/home/liuheng/edge_cloude/统计目录/realtime_escort_nocollision/refine_data2.jsonl"
DEFAULT_OUTPUT_DIR = "/home/liuheng/edge_cloude/fixed_ratio_experiments/data"
DEFAULT_TOKENIZER = "/data3/liuyuan/models/Qwen3-8B"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min_model_coverage", type=float, default=0.95)
    return parser.parse_args()


def validate_ratios(ratios: list[float]) -> list[float]:
    values = sorted(set(ratios))
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("ratios must be unique values in (0, 1]")
    return values


def read_jsonl(path: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def output_path(output_dir: Path, ratio: float) -> Path:
    return output_dir / f"fixed_ratio_{int(round(ratio * 100)):03d}.jsonl"


def load_completed(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("exact_length") is True:
                completed.add(int(row["source_index"]))
    return completed


class ExactLengthCompressor:
    def __init__(self, args: argparse.Namespace):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY in the shell before running.")
        self.client = OpenAI(api_key=api_key, base_url=args.base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        self.model = args.model
        self.temperature = args.temperature
        self.retries = args.retries
        self.min_model_coverage = args.min_model_coverage

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def force_exact(self, candidate: str, original: str, target: int) -> str:
        candidate = (candidate or "").strip()
        candidate_ids = self.encode(candidate)
        if len(candidate_ids) < target:
            candidate = candidate + "\n" + original
            candidate_ids = self.encode(candidate)

        ids = candidate_ids[:target]
        for _ in range(12):
            text = self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            actual_ids = self.encode(text)
            if len(actual_ids) == target:
                return text
            if len(actual_ids) > target:
                ids = actual_ids[:target]
                continue
            missing = target - len(actual_ids)
            filler = self.encode("\n" + original)
            ids = actual_ids + filler[:missing]
        raise RuntimeError(f"could not construct exactly {target} tokens")

    def compress(self, original: str, ratio: float) -> tuple[str, dict[str, Any]]:
        original_tokens = self.count(original)
        target = max(1, int(round(original_tokens * ratio)))
        generation_floor = target + max(24, target // 4)
        minimum_characters = max(1, int(round(len(original) * ratio * 1.20)))
        last_error: Exception | None = None
        usage: dict[str, Any] = {}
        previous_candidate = ""
        previous_tokens = 0

        for attempt in range(1, self.retries + 1):
            correction = ""
            if previous_candidate:
                correction = (
                    f"\n\nThe previous candidate had only {previous_tokens} Qwen3-8B tokens, "
                    f"below the required {target}. Rewrite it with substantially more concrete "
                    "spatial facts, constraint checks, and intermediate reasoning restored from the original. "
                    f"It must contain at least {target + max(48, target // 2)} Qwen3-8B tokens.\n"
                    f"Previous candidate:\n{previous_candidate}"
                )
            prompt = (
                "Compress the following chain of thought while preserving the facts, "
                "spatial relations, constraints, and reasoning needed to reach the same decision. "
                f"The compressed_cot value will be hard-limited to exactly {target} tokens by the Qwen3-8B tokenizer. "
                f"Write at least {generation_floor} tokens and at least {minimum_characters} Unicode characters. "
                "It is preferable to be longer because the result will be deterministically trimmed. "
                "Put the most decision-critical reasoning first. Return one JSON object with exactly one "
                "string field named compressed_cot. The field must contain only the compressed task reasoning; "
                "never discuss this instruction, compression, token counts, or the user, and do not include an action JSON.\n\n"
                f"Original chain of thought:\n{original}{correction}"
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise reasoning-compression engine."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=max(384, min(4096, target * 3 + 256)),
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                if response.choices[0].finish_reason != "stop":
                    raise RuntimeError(
                        f"incomplete API response: finish_reason={response.choices[0].finish_reason}"
                    )
                message = response.choices[0].message
                content = message.content or ""
                if not content.strip():
                    raise RuntimeError("API returned empty final content")
                payload = json.loads(content)
                compressed = payload.get("compressed_cot")
                if not isinstance(compressed, str) or not compressed.strip():
                    raise RuntimeError("API response has no non-empty compressed_cot string")
                contamination = re.compile(
                    r"(?i)\b(we need|i need|the user|user asks|compress(?:ion|ing|ed)?|"
                    r"chain of thought|hard[- ]limit(?:ed)?|token count|output only)\b"
                )
                if contamination.search(compressed):
                    raise RuntimeError("compressed_cot contains meta-instruction language")
                previous_candidate = compressed
                previous_tokens = self.count(compressed)
                model_coverage = min(1.0, previous_tokens / target)
                if model_coverage < self.min_model_coverage:
                    raise RuntimeError(
                        f"model coverage is too low: target={target}, actual={previous_tokens}, "
                        f"coverage={model_coverage:.3f}, required={self.min_model_coverage:.3f}"
                    )
                exact = self.force_exact(compressed, original, target)
                actual = self.count(exact)
                if actual != target:
                    raise RuntimeError(f"length check failed: target={target}, actual={actual}")
                if response.usage is not None:
                    usage = response.usage.model_dump()
                return exact, {
                    "attempts": attempt,
                    "original_tokens": original_tokens,
                    "target_tokens": target,
                    "actual_tokens": actual,
                    "actual_ratio": actual / original_tokens,
                    "model_output_tokens": previous_tokens,
                    "model_coverage": model_coverage,
                    "used_original_fill": previous_tokens < target,
                    "api_usage": usage,
                }
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) == 402:
                    raise RuntimeError("DeepSeek account has insufficient balance") from exc
                if attempt < self.retries:
                    time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"compression failed after {self.retries} attempts: {last_error}")


def main() -> int:
    args = parse_args()
    if not 0.0 < args.min_model_coverage <= 1.0:
        raise ValueError("min_model_coverage must be in (0, 1]")
    ratios = validate_ratios(args.ratios)
    rows = read_jsonl(args.input, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compressor = ExactLengthCompressor(args)
    write_lock = Lock()

    jobs: list[tuple[int, float, dict[str, Any]]] = []
    paths = {ratio: output_path(output_dir, ratio) for ratio in ratios}
    completed = {ratio: load_completed(path) for ratio, path in paths.items()}
    for index, row in enumerate(rows):
        for ratio in ratios:
            if index not in completed[ratio]:
                jobs.append((index, ratio, row))

    print(f"source_records={len(rows)} pending_jobs={len(jobs)} model={args.model}", flush=True)

    def run_job(job: tuple[int, float, dict[str, Any]]) -> tuple[int, float, dict[str, Any]]:
        index, ratio, row = job
        original = str(row.get("cot") or row.get("cot0") or "").strip()
        if not original:
            raise ValueError(f"source_index={index} has no cot")
        compressed, meta = compressor.compress(original, ratio)
        output = dict(row)
        output.update(
            {
                "source_index": index,
                "original_cot": original,
                "fixed_cot": compressed,
                "requested_ratio": ratio,
                "tokenizer_path": args.tokenizer,
                "compression_model": args.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "exact_length": True,
                **meta,
            }
        )
        return index, ratio, output

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(run_job, job): job[:2] for job in jobs}
        for done_count, future in enumerate(as_completed(future_map), 1):
            index, ratio = future_map[future]
            try:
                _, _, output = future.result()
                line = json.dumps(output, ensure_ascii=False) + "\n"
                with write_lock, paths[ratio].open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
            except Exception as exc:
                failures.append({"source_index": index, "ratio": ratio, "error": str(exc)})
                print(f"FAILED index={index} ratio={ratio:.0%}: {exc}", file=sys.stderr, flush=True)
            if done_count % 20 == 0 or done_count == len(jobs):
                print(f"completed={done_count}/{len(jobs)} failures={len(failures)}", flush=True)

    if failures:
        failure_path = output_dir / "failures.jsonl"
        with failure_path.open("a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
