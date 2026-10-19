import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# 默认使用 0,1 号 GPU，如需修改请在外部设置 CUDA_VISIBLE_DEVICES。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

DEFAULT_QWEN05_PATH = "/data1/liuyuan/models/Qwen2.5-0.5B-Instruct"
DEFAULT_QWEN7B_PATH = "/data1/liuyuan/models/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_REMOTE_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "model_name": DEFAULT_DEEPSEEK_MODEL,
        "kind": "remote",
        "tokenizer_path": None,
    },
    "qwen0.5b": {
        "model_name": "Qwen2.5-0.5B-Instruct",
        "kind": "local",
        "tokenizer_path": DEFAULT_QWEN05_PATH,
    },
    "qwen7b": {
        "model_name": "Qwen2.5-7B-Instruct",
        "kind": "local",
        "tokenizer_path": DEFAULT_QWEN7B_PATH,
    },
}


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _open_log_file(path: str):
    ensure_dir(os.path.dirname(path))
    try:
        return open(path, "a", encoding="utf-8", buffering=1)
    except Exception as exc:
        print(f"无法打开日志文件 {path}: {exc}", flush=True)
        return None


def _write_log_json(fh, payload: Dict[str, Any]) -> None:
    if fh is None:
        return
    try:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fh.flush()
    except Exception as exc:
        print(f"写入日志失败: {exc}", flush=True)


class SimpleProgress:
    def __init__(self, total: int, desc: str, leave: bool = False):
        self.total = max(1, total)
        self.desc = desc
        self.leave = leave
        self.current = 0
        self.last_percent = -1
        self._render(force=True)

    def _render(self, force: bool = False) -> None:
        percent = int((self.current / self.total) * 100)
        if not force and percent == self.last_percent:
            return
        self.last_percent = percent
        bar_len = 20
        filled = int(bar_len * self.current / self.total)
        bar = "#" * filled + "." * (bar_len - filled)
        print(
            f"{self.desc} [{bar}] {self.current}/{self.total} ({percent}%)",
            end="\r",
            flush=True,
        )

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        self._render()

    def set_postfix_str(self, _: str) -> None:
        # API compatibility with tqdm; no-op for the lightweight fallback.
        return

    def close(self) -> None:
        if not self.leave:
            print(" " * 80, end="\r")
        print(f"{self.desc} 完成 {self.current}/{self.total}", flush=True)


def create_progress(total: int, desc: str, enable: bool = True, leave: bool = False):
    if not enable:
        return None
    if tqdm is not None:
        return tqdm(total=total, desc=desc, ncols=90, leave=leave)
    return SimpleProgress(total, desc, leave=leave)


def log_cot_step(fh, step_record: Dict[str, Any]) -> None:
    payload = {
        "ts": time.time(),
        "model_id": step_record.get("model_id"),
        "round_id": step_record.get("round_id"),
        "step_id": step_record.get("step_id"),
        "cot_text": step_record.get("cot_text", ""),
        "actions": step_record.get("actions"),
        "action_json_text": step_record.get("action_json_text"),
        "state": step_record.get("state"),
        "parse_ok": step_record.get("llm_parse_ok"),
        "fallback_used": step_record.get("fallback_used"),
        "step_infer_time_sec": step_record.get("step_infer_time_sec"),
        "error": step_record.get("error"),
    }
    for key in [
        "llm_response_preview",
        "llm_first_parse_ok",
        "llm_first_error",
        "llm_retry_used",
        "llm_retry_parse_ok",
        "llm_retry_error",
        "llm_retry_error_detail",
        "llm_retry_response_preview",
    ]:
        if step_record.get(key) is not None:
            payload[key] = step_record.get(key)
    _write_log_json(fh, payload)


def log_progress_line(
    fh,
    model_id: str,
    round_idx: int,
    total_rounds: int,
    round_summary: Any,
    round_summaries: List[Any],
    total_steps: int,
    fallback_steps: int,
    parse_ok_steps: int,
) -> None:
    success_count = sum(1 for r in round_summaries if r.success)
    avg_time = (
        sum(r.round_time_sec for r in round_summaries) / len(round_summaries)
        if round_summaries
        else 0.0
    )
    payload = {
        "ts": time.time(),
        "model_id": model_id,
        "round_done": round_idx,
        "total_rounds": total_rounds,
        "success_rate": success_count / len(round_summaries) if round_summaries else 0.0,
        "success_count": success_count,
        "last_round_success": round_summary.success,
        "last_round_steps": round_summary.steps_used,
        "avg_round_time_sec_so_far": avg_time,
        "total_steps_so_far": total_steps,
        "fallback_steps_so_far": fallback_steps,
        "fallback_rate_so_far": (fallback_steps / total_steps) if total_steps else 0.0,
        "parse_ok_steps_so_far": parse_ok_steps,
        "parse_ok_rate_so_far": (parse_ok_steps / total_steps) if total_steps else 0.0,
    }
    _write_log_json(fh, payload)


class TokenCounter:
    def __init__(self, model_id: str, tokenizer_path: Optional[str] = None):
        self.model_id = model_id
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.fallback_encoder = None
        self.backend = None
        self.warning = None
        self.token_count_fallback = False
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        if self.tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                self.backend = f"transformers:{self.tokenizer_path}"
                return
            except Exception as exc:
                self.warning = f"transformers tokenizer load failed: {exc}"

        if tiktoken is not None:
            try:
                self.fallback_encoder = tiktoken.get_encoding("cl100k_base")
                self.backend = "tiktoken/cl100k_base"
            except Exception as exc:
                self.warning = f"tiktoken unavailable: {exc}"
        self.token_count_fallback = True
        if self.backend is None:
            self.backend = "len(text_bytes)"

    def count(self, text: str) -> Tuple[int, Dict[str, Any]]:
        if text is None:
            text = ""
        token_count = 0
        if self.tokenizer is not None:
            try:
                token_count = len(
                    self.tokenizer.encode(text, add_special_tokens=False)
                )
            except Exception as exc:
                self.warning = f"tokenizer.encode failed: {exc}"
                token_count = 0
        elif self.fallback_encoder is not None:
            try:
                token_count = len(self.fallback_encoder.encode(text))
            except Exception as exc:
                self.warning = f"tiktoken.encode failed: {exc}"
                token_count = len(text.encode("utf-8"))
        else:
            token_count = len(text.encode("utf-8"))

        meta = {
            "token_counter_backend": self.backend,
            "token_count_fallback": self.token_count_fallback,
        }
        if self.warning:
            meta["warning"] = self.warning
        return token_count, meta


class LLMRunner:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        base_url: str,
        api_key: str,
        backend: str = "openai",
        tokenizer_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = backend
        self.api_key = api_key or "EMPTY"
        self.base_url = base_url
        self.tokenizer_path = tokenizer_path
        self.client: Optional[OpenAI] = None
        self.hf_model = None
        self.hf_tokenizer = None

        if backend == "openai":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unsupported backend {backend}")

    def _init_transformers(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except Exception as exc:
            raise ValueError(
                "Transformers backend requested but transformers is unavailable"
            ) from exc

        model_source = self.tokenizer_path or self.model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_source)
        pad_added = False
        if self.hf_tokenizer.pad_token_id is None:
            new_pad = "<|pad|>"
            self.hf_tokenizer.add_special_tokens({"pad_token": new_pad})
            pad_added = True
        if self.hf_tokenizer.pad_token_id is None and self.hf_tokenizer.eos_token is not None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        pad_id = self.hf_tokenizer.pad_token_id or self.hf_tokenizer.eos_token_id
        self.hf_tokenizer.padding_side = "left"

        load_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        try:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_source,
                dtype=torch.float16,
                **load_kwargs,
            )
        except TypeError:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                **load_kwargs,
            )
        if pad_added:
            self.hf_model.resize_token_embeddings(len(self.hf_tokenizer))
        self.hf_model.config.pad_token_id = pad_id
        self._transformers_pad_id = pad_id

    def infer(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], float]:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if self.backend == "openai":
            start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start
            text = (response.choices[0].message.content or "").strip()
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(
                    response.usage, "completion_tokens", None
                ),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
            return text, usage, elapsed

        return self._infer_transformers(
            messages, temperature=temperature, max_tokens=max_tokens
        )

    def _infer_transformers(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any], float]:
        if self.hf_model is None or self.hf_tokenizer is None:
            raise ValueError("Transformers backend not initialized")

        import torch

        inputs = None
        if hasattr(self.hf_tokenizer, "apply_chat_template"):
            try:
                prompt = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.hf_tokenizer(prompt, return_tensors="pt")
            except Exception:
                inputs = None

        if inputs is None:
            prompt = "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in messages]
            )
            inputs = self.hf_tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        if torch.cuda.is_available():
            model_input_device = None
            if hasattr(self.hf_model, "hf_device_map"):
                for dev in self.hf_model.hf_device_map.values():
                    if isinstance(dev, int):
                        model_input_device = torch.device(f"cuda:{dev}")
                        break
                    if isinstance(dev, str) and dev.startswith("cuda"):
                        model_input_device = torch.device(dev)
                        break
            if model_input_device is None:
                model_input_device = torch.device("cuda:0")
            input_ids = input_ids.to(model_input_device)
            attention_mask = attention_mask.to(model_input_device)

        start = time.perf_counter()
        with torch.no_grad():
            eos_token_id = getattr(self.hf_tokenizer, "eos_token_id", None)
            output_ids = self.hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                max_new_tokens=max_tokens,
                pad_token_id=self._transformers_pad_id,
                eos_token_id=eos_token_id,
            )
        elapsed = time.perf_counter() - start
        generated = output_ids[0, input_ids.shape[1] :]
        text = self.hf_tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, {}, elapsed


def extract_actions_from_text(
    response_text: str,
) -> Tuple[Optional[List[int]], str, str, bool, bool, Optional[str]]:
    """Parse model output into actions and thought.

    兼容以下格式：
    - ```json ... ``` / ``` ... ``` 代码块中的 JSON
    - 直接输出 JSON（不带代码块）
    - hunter0/hunter1/hunter2 的键值对（非 JSON），尽量解析
    """

    def extract_balanced_json(text: str, start_idx: int) -> Optional[str]:
        if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
        return None

    def coerce_action(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid action")
        if isinstance(value, int):
            if 0 <= value <= 4:
                return value
            raise ValueError(f"action out of range: {value}")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"non-integer action value: {value}")
            as_int = int(value)
            if 0 <= as_int <= 4:
                return as_int
            raise ValueError(f"action out of range: {as_int}")
        if isinstance(value, str):
            s = value.strip().lower()
            if re.fullmatch(r"-?\d+", s):
                as_int = int(s)
                if 0 <= as_int <= 4:
                    return as_int
                raise ValueError(f"action out of range: {as_int}")
            mapping = {
                "stay": 0,
                "停留": 0,
                "无动作": 0,
                "不动": 0,
                "up": 1,
                "上": 1,
                "down": 2,
                "下": 2,
                "left": 3,
                "左": 3,
                "right": 4,
                "右": 4,
            }
            if s in mapping:
                return mapping[s]
        raise ValueError(f"invalid action value: {value!r}")

    def find_key(obj: Any, key: str) -> Optional[Any]:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = find_key(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = find_key(v, key)
                if found is not None:
                    return found
        return None

    def try_parse_json(text: str) -> Tuple[Optional[List[int]], Optional[str]]:
        try:
            parsed = json.loads(text)
        except Exception as exc:
            return None, str(exc)
        v0 = find_key(parsed, "hunter0")
        v1 = find_key(parsed, "hunter1")
        v2 = find_key(parsed, "hunter2")
        if v0 is None or v1 is None or v2 is None:
            return None, "missing hunter keys"
        try:
            return [coerce_action(v0), coerce_action(v1), coerce_action(v2)], None
        except Exception as exc:
            return None, str(exc)

    response_text = response_text or ""
    action_json_text = ""
    parse_ok = False
    fallback_used = False
    error: Optional[str] = None

    # thought：对齐 111 的逻辑，默认取代码块之前的文本
    thought = response_text.strip()
    if "```" in response_text:
        thought = response_text.split("```")[0].strip()

    candidates: List[str] = []

    # 1) 优先解析代码块（对齐 111 的输出习惯）
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", response_text, re.S)
    for block in reversed(fenced_blocks):
        block = block.strip()
        if block:
            candidates.append(block)

    # 2) 若无代码块，尝试从原文本中提取平衡括号的 JSON
    lowered = response_text.lower()
    key_pos = lowered.rfind("hunter0")
    if key_pos != -1:
        start = response_text.rfind("{", 0, key_pos)
        if start != -1:
            extracted = extract_balanced_json(response_text, start)
            if extracted:
                candidates.append(extracted.strip())

    # 3) 直接就是 JSON 的情况
    stripped = response_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    actions: Optional[List[int]] = None

    for cand in candidates:
        cand_strip = cand.strip()
        if "{" in cand_strip:
            start_idx = cand_strip.find("{")
            if start_idx != -1 and not cand_strip.strip().startswith("{"):
                cand_strip = cand_strip[start_idx:]
            if cand_strip.startswith("{") and not cand_strip.endswith("}"):
                extracted = extract_balanced_json(cand_strip, 0)
                if extracted:
                    cand_strip = extracted

        for variant in [cand_strip, cand_strip.replace("'", '"')]:
            if not variant.strip().startswith("{"):
                continue
            parsed_actions, parse_err = try_parse_json(variant.strip())
            if parsed_actions is not None:
                actions = parsed_actions
                action_json_text = variant.strip()
                parse_ok = True
                error = None
                break
            error = parse_err
        if parse_ok:
            break

    # 4) 最后兜底：从文本里抽取键值对（非 JSON 也尽量解析）
    if actions is None:
        def _extract_hunter_action(hunter_idx: int) -> Optional[int]:
            # 1) hunter0: 0 / hunter0 = 0
            m = re.search(
                rf"hunter{hunter_idx}\s*[:=]\s*([0-4])", response_text, re.I
            )
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None

            # 2) hunter0: up / hunter0: 左
            m = re.search(
                rf"hunter{hunter_idx}\s*[:=]\s*(stay|up|down|left|right|停留|上|下|左|右)",
                response_text,
                re.I,
            )
            if m:
                try:
                    return coerce_action(m.group(1))
                except Exception:
                    return None

            # 3) Hunter 0: 上(0) / **Hunter 0**: right ...
            m = re.search(
                rf"(?:\*\*\s*)?Hunter\s*{hunter_idx}(?:\s*\*\*)?\s*[:：][^\n]*?"
                rf"(stay|up|down|left|right|停留|上|下|左|右)",
                response_text,
                re.I,
            )
            if m:
                try:
                    return coerce_action(m.group(1))
                except Exception:
                    return None

            return None

        h0 = _extract_hunter_action(0)
        h1 = _extract_hunter_action(1)
        h2 = _extract_hunter_action(2)
        if h0 is not None and h1 is not None and h2 is not None:
            actions = [h0, h1, h2]
            action_json_text = json.dumps(
                {"hunter0": actions[0], "hunter1": actions[1], "hunter2": actions[2]},
                ensure_ascii=False,
            )
            parse_ok = True
            error = None

    if actions is None:
        fallback_used = True

    return actions, thought, action_json_text, parse_ok, fallback_used, error


@dataclass
class RoundSummary:
    model_id: str
    round_id: int
    success: bool
    steps_used: int
    round_time_sec: float
    round_cot_tokens_sum: int
    round_total_tokens_sum: int
    avg_tokens_per_step: float
    avg_step_infer_time_sec: float
    token_count_fallback: bool
    usage_prompt_tokens_sum: Optional[int]
    usage_completion_tokens_sum: Optional[int]
    usage_total_tokens_sum: Optional[int]


class HuntingGame:
    def __init__(
        self,
        model_id: str,
        runner: LLMRunner,
        token_counter: TokenCounter,
        grid_size: int = 20,
        max_steps: int = 100,
        visualize: bool = False,
        # COT精炼实验参数（仅对 deepseek 有效；其他模型会忽略）
        cot_refine_enabled: bool = False,
        cot_refine_start: float = 0.30,
        cot_refine_step: float = 0.01,
        cot_refine_max: float = 1.00,
        cot_refine_temperature: float = 0.0,
        cot_refine_action_temperature: float = 0.0,
        cot_refine_action_max_tokens: int = 128,
    ):
        self.model_id = model_id
        self.runner = runner
        self.token_counter = token_counter
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.visualize = visualize

        # COT精炼实验配置
        self.cot_refine_enabled = bool(cot_refine_enabled) and (model_id == "deepseek")
        self.cot_refine_start = float(cot_refine_start)
        self.cot_refine_step = float(cot_refine_step)
        self.cot_refine_max = float(cot_refine_max)
        self.cot_refine_temperature = float(cot_refine_temperature)
        self.cot_refine_action_temperature = float(cot_refine_action_temperature)
        self.cot_refine_action_max_tokens = int(cot_refine_action_max_tokens)

        positions = random.sample(
            [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)], 4
        )
        self.hunters = positions[:3]
        self.prey = positions[3]
        self.current_step = 0

        self.action_map = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
        }
        self.action_names = ["停留", "上", "下", "左", "右"]

        self.history = {
            "hunter_0": [{"step": 0, "position": self.hunters[0], "action": None}],
            "hunter_1": [{"step": 0, "position": self.hunters[1], "action": None}],
            "hunter_2": [{"step": 0, "position": self.hunters[2], "action": None}],
            "prey": [{"step": 0, "position": self.prey, "action": None}],
        }

        self.fig = None
        self.ax = None
        self.hunter_scat = None
        self.prey_scat = None
        self.frames: List[np.ndarray] = []

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_inside_triangle(self, point, triangle):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(point, triangle[0], triangle[1])
        d2 = sign(point, triangle[1], triangle[2])
        d3 = sign(point, triangle[2], triangle[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def check_win(self):
        distances = [self.manhattan_distance(h, self.prey) for h in self.hunters]
        if all(d < 2 for d in distances):
            if self.is_inside_triangle(self.prey, self.hunters):
                return True
        return False

    def move_agent(self, position, action, occupied_positions=None):
        delta = self.action_map[action]
        new_pos = (position[0] + delta[0], position[1] + delta[1])
        new_pos = (
            max(0, min(self.grid_size - 1, new_pos[0])),
            max(0, min(self.grid_size - 1, new_pos[1])),
        )
        if occupied_positions is not None and new_pos in occupied_positions:
            return position
        return new_pos

    def get_state_description(self):
        return {
            "step": self.current_step,
            "hunters": [{"id": i, "position": list(self.hunters[i])} for i in range(3)],
            "prey": {"position": list(self.prey)},
            "distances": [
                self.manhattan_distance(self.hunters[i], self.prey) for i in range(3)
            ],
        }

    def get_prompt(self) -> str:
        state = self.get_state_description()
        prompt = f"""你正在控制三个猎人（Hunter 0, 1, 2）在{self.grid_size}x{self.grid_size}的网格中围捕一个猎物。

当前状态：
- 步数: {state['step']}/{self.max_steps}
- 猎人位置: 
  Hunter 0: {state['hunters'][0]['position']} (距离猎物: {state['distances'][0]})
  Hunter 1: {state['hunters'][1]['position']} (距离猎物: {state['distances'][1]})
  Hunter 2: {state['hunters'][2]['position']} (距离猎物: {state['distances'][2]})
- 猎物位置: {state['prey']['position']}

决策目标：
* 控制三名猎人，使它们逐步减少与猎物的平均距离，并封锁猎物的可逃路径。
* 每个猎人的决策相互独立，但应整体朝向包围猎物的方向。

决策规则：
1. 对于每个猎人，从 0,1,2,3,4 五个动作编号中选取一个（0=停留, 1=上, 2=下, 3=左, 4=右）。
2. 优先规则如下：
   * 优先选择能减少猎人与猎物距离的动作。
   * 若多个动作使距离相同或相等，选择第一个满足条件的动作（0→1→2→3→4 的顺序）。
   * 若所有动作都会增加距离，则选择第一个动作（0）。
3. 如果该猎人只有一个可行动作（例如边界限制），则直接选择该动作。
4. 所有猎人决策完成后，返回三名猎人的动作结果。

输出要求：
- 请你输出自己的思考过程，用有序列表，每条为一句可读的推理步骤与思考方式，思考过程不要有直接的动作结果，同时不要有多余的输出。
- 思考过程禁止写出具体动作编号或“选择动作X/方向Y”之类的决策表述，只描述分析推理。
- 然后以 JSON 格式返回动作结果，注意输出的json格式必须要用符号```包裹起来。
- 严禁输出 JSON 数组（例如 `[...]`），严禁输出 `action/reason` 这类字段；最终 JSON 必须是对象，且只能包含 `hunter0/hunter1/hunter2` 三个键，值必须是 0~4 的整数。
- 输出格式如下（必须严格一致）：
```json
{{"hunter0": 0, "hunter1": 0, "hunter2": 0}}
```
"""
        return prompt

    def _build_system_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                f"你是一个围捕游戏的智能体控制器，负责在 {self.grid_size}x{self.grid_size} 网格中控制三个猎人（Hunter 0, 1, 2）去围捕一个猎物。"
                f"动作编码：0=停留, 1=上, 2=下, 3=左, 4=右。最大步数 {self.max_steps}。"
            ),
        }

    def _refine_cot_once(
        self,
        original_cot: str,
        ratio: float,
        original_cot_tokens: int,
    ) -> Tuple[str, Dict[str, Any], float, int]:
        """用云端模型把 original_cot 精炼到约 ratio 的 token 数。返回(精炼cot, usage, infer_time, 精炼cot_tokens)。"""
        ratio = max(0.01, min(1.0, float(ratio)))
        target_tokens = max(1, int(round(original_cot_tokens * ratio)))

        sys_msg = {
            "role": "system",
            "content": (
                "你是一个COT精炼器。你的任务是将给定的思维链(COT)压缩/精炼到指定比例，"
                "但必须尽量保留支撑最终动作决策所需的信息、关键推理步骤、以及对三名猎人动作选择有影响的细节。"
                "只输出精炼后的COT文本，不要输出JSON，不要输出解释，不要添加标题。"
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"压缩比例(保留token比例)：{ratio:.2%}\\n"
                f"原始COT token数：{original_cot_tokens}\\n"
                f"目标token数(约)：{target_tokens}\\n\\n"
                "要求：\\n"
                "1) 只能输出精炼后的COT正文；不要输出任何其它文字。\\n"
                "2) 尽量让精炼后的COT长度接近目标token数(允许±5%)。\\n"
                "3) 必须保留对三名猎人最终动作选择有贡献的关键信息，避免遗漏导致动作改变。\\n"
                "4) 不要在精炼文本中直接写出最终动作编号/方向或“选择动作X”之类的决策表述；保留推理依据即可。\\n\\n"
                f"原始COT：\\n{original_cot}"
            ),
        }

        max_tokens = min(self.runner.max_tokens, max(128, target_tokens + 128))
        refined_text, usage, infer_time = self.runner.infer(
            [sys_msg, user_msg],
            temperature=self.cot_refine_temperature,
            max_tokens=max_tokens,
        )
        refined_text = (refined_text or "").strip()
        refined_tokens, _ = self.token_counter.count(refined_text)
        return refined_text, usage or {}, infer_time, refined_tokens

    def _infer_actions_with_refined_cot(
        self,
        prompt: str,
        refined_cot: str,
    ) -> Tuple[Optional[List[int]], Dict[str, Any], float, bool, Optional[str]]:
        """把 refined_cot 当作“上一轮思考过程”，再次让模型只输出动作 JSON。"""
        sys_msg = self._build_system_message()
        # 给模型同样的状态提示，但强调不要重新推理
        # user1 = {
        #     "role": "user",
        #     "content": (
        #         prompt
        #         + "\\n\\n注意：接下来会提供你上一轮的思考过程（精炼版）。请尽量沿用其中隐含的决策，不要重新推理。"
        #     ),
        # }
        # assistant_thought = {"role": "assistant", "content": refined_cot}
        # user2 = {
        #     "role": "user",
        #     "content": (
        #         "请严格只输出最终动作 JSON（对象格式，键为 hunter0/hunter1/hunter2，值为0-4）。"
        #         "不要输出任何解释或多余文本。"
        #     ),
        # }
        user1 = {
            "role": "user",
            "content": "以下是之前的思考过程（精炼版）："
        }
        assistant_thought = {"role": "assistant", "content": refined_cot}
        user2 = {
            "role": "user",
            "content": (
            "请**严格基于以上思考过程**，直接输出三个猎人的动作决策。\n"
            "注意：不要重新分析状态，不要重新推理，只根据上述思考输出动作。\n"
            "输出格式：{\"hunter0\": <0-4>, \"hunter1\": <0-4>, \"hunter2\": <0-4>}"
            ),
        }
        text, usage, infer_time = self.runner.infer(
            [sys_msg, user1, assistant_thought, user2],
            temperature=self.cot_refine_action_temperature,
            max_tokens=min(self.runner.max_tokens, int(self.cot_refine_action_max_tokens)),
        )
        actions, _thought, _action_json_text, parse_ok, _fallback_used, parse_error = extract_actions_from_text(text)
        return actions, usage or {}, infer_time, bool(parse_ok), parse_error

    def _search_matching_refined_cot(
        self,
        prompt: str,
        original_cot: str,
        original_actions: List[int],
    ) -> Dict[str, Any]:
        """从 cot_refine_start 开始逐步增加比例，直到“精炼cot->动作”与原始动作一致。"""
        original_cot_tokens, _ = self.token_counter.count(original_cot)
        ratio = self.cot_refine_start
        attempts = 0
        total_refine_calls = 0
        total_action_calls = 0
        total_infer_time = 0.0

        final: Dict[str, Any] = {
            "refine_enabled": True,
            "refine_start_ratio": self.cot_refine_start,
            "refine_step": self.cot_refine_step,
            "refine_max_ratio": self.cot_refine_max,
            "original_cot_tokens": original_cot_tokens,
            "final_ratio": None,
            "match": False,
            "refined_cot_text": None,
            "refined_cot_tokens": None,
            "refined_actions": None,
            "attempts": 0,
            "refine_total_infer_time_sec": 0.0,
        }

        while ratio <= self.cot_refine_max + 1e-9:
            attempts += 1
            refined_cot, refine_usage, refine_time, refined_tokens = self._refine_cot_once(
                original_cot=original_cot,
                ratio=ratio,
                original_cot_tokens=original_cot_tokens,
            )
            total_refine_calls += 1
            total_infer_time += refine_time

            test_actions, action_usage, action_time, parse_ok, parse_error = self._infer_actions_with_refined_cot(
                prompt=prompt,
                refined_cot=refined_cot,
            )
            total_action_calls += 1
            total_infer_time += action_time

            match = (test_actions == original_actions) if test_actions is not None else False
            if match:
                final.update(
                    {
                        "final_ratio": ratio,
                        "match": True,
                        "refined_cot_text": refined_cot,
                        "refined_cot_tokens": refined_tokens,
                        "refined_actions": test_actions,
                        "attempts": attempts,
                        "refine_total_infer_time_sec": total_infer_time,
                        "refine_total_refine_calls": total_refine_calls,
                        "refine_total_action_calls": total_action_calls,
                        "refine_last_refine_usage": refine_usage,
                        "refine_last_action_usage": action_usage,
                    }
                )
                return final

            # mismatch: 增加比例（保留更多token）
            ratio = round(ratio + self.cot_refine_step, 10)

        # 未找到匹配
        final.update(
            {
                "final_ratio": None,
                "match": False,
                "attempts": attempts,
                "refine_total_infer_time_sec": total_infer_time,
                "refine_total_refine_calls": total_refine_calls,
                "refine_total_action_calls": total_action_calls,
            }
        )
        return final


    def get_llm_actions(
        self, round_id: int, step_id: int
    ) -> Tuple[List[int], Dict[str, Any]]:
        state = self.get_state_description()
        prompt = self.get_prompt()

        messages = [
            {
                "role": "system",
                "content": f"你是一个围捕游戏的智能体控制器，负责在 {self.grid_size}x{self.grid_size} 网格中控制三个猎人（Hunter 0, 1, 2）去围捕一个猎物。动作编码：0=停留, 1=上, 2=下, 3=左, 4=右。最大步数 {self.max_steps}",
            },
            {"role": "user", "content": prompt},
        ]

        usage_meta: Dict[str, Any] = {}
        infer_time_total = 0.0

        first_response_text = ""
        try:
            first_response_text, usage_meta, infer_time = self.runner.infer(messages)
            infer_time_total += infer_time
        except Exception as exc:
            usage_meta["error"] = str(exc)
            first_response_text = f"LLM 调用失败: {exc}"

        first_response_len = len(first_response_text or "")
        first_response_preview = (first_response_text or "").strip()
        if len(first_response_preview) > 500:
            first_response_preview = first_response_preview[:500] + "...[truncated]"

        first_actions, first_thought, first_action_json_text, first_parse_ok, first_fallback_used, first_parse_error = (
            extract_actions_from_text(first_response_text)
        )

        # 默认使用第一次输出；若解析失败，再进行一次“只输出 JSON”的低成本重试，避免浪费 token。
        actions = first_actions
        thought = first_thought
        action_json_text = first_action_json_text
        parse_ok = first_parse_ok
        fallback_used = first_fallback_used
        parse_error = first_parse_error

        retry_used = False
        retry_error: Optional[str] = None
        retry_response_preview: Optional[str] = None
        retry_parse_ok: Optional[bool] = None
        retry_parse_error: Optional[str] = None
        if actions is None:
            retry_used = True
            retry_prompt = (
                "你只需要输出最终动作 JSON，不需要思考过程，不需要解释，不需要任何多余文字。\n"
                "动作编码：0=停留, 1=上, 2=下, 3=左, 4=右。\n"
                "必须严格输出如下格式（注意是对象不是数组）：\n"
                "```json\n"
                "{\"hunter0\": 0, \"hunter1\": 0, \"hunter2\": 0}\n"
                "```\n\n"
                f"当前状态：\n"
                f"- 步数: {state['step']}/{self.max_steps}\n"
                f"- Hunter 0: {state['hunters'][0]['position']} (距离猎物: {state['distances'][0]})\n"
                f"- Hunter 1: {state['hunters'][1]['position']} (距离猎物: {state['distances'][1]})\n"
                f"- Hunter 2: {state['hunters'][2]['position']} (距离猎物: {state['distances'][2]})\n"
                f"- 猎物位置: {state['prey']['position']}\n"
            )
            retry_messages = [
                messages[0],
                {"role": "user", "content": retry_prompt},
            ]
            try:
                retry_text, retry_usage, retry_time = self.runner.infer(
                    retry_messages,
                    temperature=0.0,
                    max_tokens=min(self.runner.max_tokens, 64),
                )
                infer_time_total += retry_time
                if retry_usage:
                    usage_meta.update(
                        {f"retry_{k}": v for k, v in retry_usage.items() if v is not None}
                    )
                retry_response_preview = (retry_text or "").strip()
                if retry_response_preview and len(retry_response_preview) > 500:
                    retry_response_preview = retry_response_preview[:500] + "...[truncated]"
                actions2, thought2, action_json_text2, parse_ok2, fallback_used2, parse_error2 = (
                    extract_actions_from_text(retry_text)
                )
                retry_parse_ok = parse_ok2
                retry_parse_error = parse_error2
                if actions2 is not None:
                    actions = actions2
                    action_json_text = action_json_text2
                    parse_ok = parse_ok2
                    fallback_used = fallback_used2
                    parse_error = parse_error2
                    # 保留第一次输出的思考过程（若有），避免二次请求只返回 JSON 导致 cot 为空。
                    thought = first_thought
            except Exception as exc:
                retry_error = str(exc)

        error_msg = parse_error
        if actions is None:
            actions = self.get_greedy_actions()
            if not parse_ok:
                error_msg = error_msg or "parse_failed_greedy_fallback"
            action_json_text = json.dumps(
                {"hunter0": actions[0], "hunter1": actions[1], "hunter2": actions[2]},
                ensure_ascii=False,
            )

        actions = [int(max(0, min(4, int(a)))) for a in actions]

        cot_tokens, cot_meta = self.token_counter.count(thought)
        action_tokens, action_meta = self.token_counter.count(action_json_text)
        step_total_tokens = cot_tokens + action_tokens

        step_record: Dict[str, Any] = {
            "model_id": self.model_id,
            "round_id": round_id,
            "step_id": step_id,
            "cot_text": thought,
            "cot_tokens": cot_tokens,
            "cot_token_counter_backend": cot_meta.get("token_counter_backend"),
            "token_count_fallback": cot_meta.get("token_count_fallback", False),
            "action_json_text": action_json_text,
            "action_tokens": action_tokens,
            "action_token_counter_backend": action_meta.get("token_counter_backend"),
            "action_token_count_fallback": action_meta.get(
                "token_count_fallback", False
            ),
            "step_total_tokens": step_total_tokens,
            "step_infer_time_sec": infer_time_total,
            "llm_parse_ok": parse_ok,
            "fallback_used": fallback_used,
            "error": error_msg,
            "usage_prompt_tokens": usage_meta.get("prompt_tokens"),
            "usage_completion_tokens": usage_meta.get("completion_tokens"),
            "usage_total_tokens": usage_meta.get("total_tokens"),
            "state": state,
            "actions": actions,
        }
        step_record["llm_response_len"] = first_response_len
        step_record["llm_response_preview"] = first_response_preview
        step_record["llm_first_parse_ok"] = first_parse_ok
        if first_parse_error:
            step_record["llm_first_error"] = first_parse_error
        if retry_used:
            step_record["llm_retry_used"] = True
            if retry_error:
                step_record["llm_retry_error"] = retry_error
            if retry_parse_ok is not None:
                step_record["llm_retry_parse_ok"] = retry_parse_ok
            if retry_parse_error:
                step_record["llm_retry_error_detail"] = retry_parse_error
            if retry_response_preview is not None:
                step_record["llm_retry_response_preview"] = retry_response_preview
        if cot_meta.get("warning"):
            step_record["token_counter_warning"] = cot_meta["warning"]
        if usage_meta.get("error"):
            step_record["llm_error"] = usage_meta["error"]

        # ===== COT精炼实验（DeepSeek）=====
        if self.cot_refine_enabled and thought and isinstance(actions, list) and len(actions) == 3:
            try:
                refine_result = self._search_matching_refined_cot(
                    prompt=prompt,
                    original_cot=thought,
                    original_actions=actions,
                )
                step_record["cot_refine"] = refine_result
            except Exception as exc:
                step_record["cot_refine_error"] = str(exc)

        # 额外展开一层，便于后处理：坐标 & 精炼比例
        try:
            step_record["hunter_positions"] = [
                state["hunters"][0]["position"],
                state["hunters"][1]["position"],
                state["hunters"][2]["position"],
            ]
            step_record["prey_position"] = state["prey"]["position"]
            if step_record.get("cot_refine") and step_record["cot_refine"].get("match"):
                step_record["cot_refine_final_ratio"] = step_record["cot_refine"]["final_ratio"]
                step_record["cot_refine_cot_text"] = step_record["cot_refine"]["refined_cot_text"]
        except Exception:
            pass
        return actions, step_record

    def get_greedy_actions(self):
        actions = []
        occupied = set()
        for hunter in self.hunters:
            best_action = 0
            best_dist = self.manhattan_distance(hunter, self.prey)
            for action in range(5):
                new_pos = self.move_agent(hunter, action, occupied)
                dist = self.manhattan_distance(new_pos, self.prey)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action
            actions.append(best_action)
            occupied.add(self.move_agent(hunter, best_action, set()))
        return actions

    def step(self, round_id: int) -> Dict[str, Any]:
        step_id = self.current_step + 1
        hunter_actions, step_record = self.get_llm_actions(round_id, step_id)
        occupied_positions = set()
        new_hunters = []
        for i in range(3):
            new_pos = self.move_agent(self.hunters[i], hunter_actions[i], occupied_positions)
            new_hunters.append(new_pos)
            occupied_positions.add(new_pos)

        self.hunters = new_hunters
        for i in range(3):
            self.history[f"hunter_{i}"].append(
                {"step": step_id, "position": self.hunters[i], "action": hunter_actions[i]}
            )

        prey_action = random.randint(0, 4)
        self.prey = self.move_agent(self.prey, prey_action, occupied_positions)
        self.history["prey"].append({"step": step_id, "position": self.prey, "action": prey_action})

        step_record["prey_action"] = prey_action
        step_record["hunter_actions"] = hunter_actions
        self.current_step = step_id
        return step_record

    def render(self, save_frame=True):
        if plt is None:
            return
        if self.fig is None:
            import matplotlib

            matplotlib.use("Agg")
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
            self.ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
            self.ax.grid(True, color="gray", linewidth=0.5, alpha=0.6)
            self.ax.set_aspect("equal")
            self.ax.invert_yaxis()
            self.ax.set_title("Pursuit Game")

            hunter_positions = np.array([self.hunters[i] for i in range(3)])
            prey_position = np.array([self.prey])
            self.hunter_scat = self.ax.scatter(
                hunter_positions[:, 1] + 0.5,
                hunter_positions[:, 0] + 0.5,
                c="blue",
                s=200,
                label="Hunters",
                marker="o",
                edgecolors="black",
                linewidths=2,
            )
            self.prey_scat = self.ax.scatter(
                prey_position[:, 1] + 0.5,
                prey_position[:, 0] + 0.5,
                c="red",
                s=300,
                marker="*",
                label="Prey",
                edgecolors="black",
                linewidths=2,
            )
            self.ax.legend(loc="upper right", fontsize=12)
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.tick_params(left=False, bottom=False)
        else:
            hunter_positions = np.array([self.hunters[i] for i in range(3)])
            prey_position = np.array([self.prey])
            self.hunter_scat.set_offsets(
                np.c_[hunter_positions[:, 1] + 0.5, hunter_positions[:, 0] + 0.5]
            )
            self.prey_scat.set_offsets(
                np.c_[prey_position[:, 1] + 0.5, prey_position[:, 0] + 0.5]
            )

        prey_pos = self.prey
        hunter_positions = self.hunters
        distances = [abs(hx - prey_pos[0]) + abs(hy - prey_pos[1]) for hx, hy in hunter_positions]
        avg_dist = sum(distances) / len(distances)
        in_triangle = self.is_inside_triangle(prey_pos, hunter_positions)

        title = f"Step: {self.current_step} | Avg Dist: {avg_dist:.1f} | In Triangle: {in_triangle}"
        self.ax.set_title(title, fontsize=14)

        if save_frame:
            self.fig.canvas.draw()
            import io
            from PIL import Image

            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
            frame = np.array(img)[:, :, :3]
            buf.close()
            self.frames.append(frame.copy())

    def run_round(
        self,
        round_id: int,
        show_progress: bool = True,
        cot_logger=None,
    ) -> Tuple[RoundSummary, List[Dict[str, Any]]]:
        steps: List[Dict[str, Any]] = []
        round_start = time.perf_counter()
        step_bar = create_progress(
            self.max_steps,
            f"[{self.model_id}] Round {round_id} steps",
            enable=show_progress,
            leave=False,
        )
        try:
            while self.current_step < self.max_steps:
                step_record = self.step(round_id)
                steps.append(step_record)
                if cot_logger:
                    cot_logger(step_record)
                if self.visualize:
                    self.render()
                if step_bar:
                    step_bar.update(1)
                elif step_record["step_id"] % 10 == 0 or step_record["step_id"] == 1:
                    print(
                        f"[{self.model_id}] Round {round_id} step {step_record['step_id']}/{self.max_steps}",
                        flush=True,
                    )
                if self.check_win():
                    break
        finally:
            if step_bar:
                step_bar.close()

        success = self.check_win()
        steps_used = self.current_step if success else self.max_steps
        round_time = time.perf_counter() - round_start

        round_cot_tokens_sum = sum(s.get("cot_tokens", 0) for s in steps)
        round_total_tokens_sum = sum(s.get("step_total_tokens", 0) for s in steps)
        avg_tokens_per_step = (
            round_total_tokens_sum / len(steps) if steps else 0.0
        )
        avg_step_infer_time_sec = (
            sum(s.get("step_infer_time_sec", 0.0) for s in steps) / len(steps)
            if steps
            else 0.0
        )
        usage_prompt_tokens_sum = sum(
            s.get("usage_prompt_tokens") or 0 for s in steps
        ) or None
        usage_completion_tokens_sum = sum(
            s.get("usage_completion_tokens") or 0 for s in steps
        ) or None
        usage_total_tokens_sum = sum(
            s.get("usage_total_tokens") or 0 for s in steps
        ) or None

        round_summary = RoundSummary(
            model_id=self.model_id,
            round_id=round_id,
            success=success,
            steps_used=steps_used,
            round_time_sec=round_time,
            round_cot_tokens_sum=round_cot_tokens_sum,
            round_total_tokens_sum=round_total_tokens_sum,
            avg_tokens_per_step=avg_tokens_per_step,
            avg_step_infer_time_sec=avg_step_infer_time_sec,
            token_count_fallback=any(
                s.get("token_count_fallback")
                or s.get("action_token_count_fallback")
                for s in steps
            ),
            usage_prompt_tokens_sum=usage_prompt_tokens_sum,
            usage_completion_tokens_sum=usage_completion_tokens_sum,
            usage_total_tokens_sum=usage_total_tokens_sum,
        )
        return round_summary, steps


def aggregate_model_metrics(
    model_id: str, round_summaries: List[RoundSummary], step_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    num_games = len(round_summaries)
    success_count = sum(1 for r in round_summaries if r.success)
    success_steps = [r.steps_used for r in round_summaries if r.success]
    avg_success_steps = (
        sum(success_steps) / len(success_steps) if success_steps else 0.0
    )
    avg_steps_overall = (
        sum(r.steps_used for r in round_summaries) / num_games if num_games else 0.0
    )
    avg_round_time_sec = (
        sum(r.round_time_sec for r in round_summaries) / num_games if num_games else 0.0
    )
    avg_round_cot_tokens = (
        sum(r.round_cot_tokens_sum for r in round_summaries) / num_games
        if num_games
        else 0.0
    )
    avg_round_total_tokens = (
        sum(r.round_total_tokens_sum for r in round_summaries) / num_games
        if num_games
        else 0.0
    )
    total_steps_recorded = len(step_records)
    avg_step_infer_time_sec = (
        sum(s.get("step_infer_time_sec", 0.0) for s in step_records) / total_steps_recorded
        if total_steps_recorded
        else 0.0
    )
    avg_tokens_per_step = (
        sum(s.get("step_total_tokens", 0) for s in step_records) / total_steps_recorded
        if total_steps_recorded
        else 0.0
    )
    token_count_fallback = any(r.token_count_fallback for r in round_summaries)

    usage_prompt_tokens = [
        r.usage_prompt_tokens_sum for r in round_summaries if r.usage_prompt_tokens_sum
    ]
    usage_completion_tokens = [
        r.usage_completion_tokens_sum
        for r in round_summaries
        if r.usage_completion_tokens_sum
    ]
    usage_total_tokens = [
        r.usage_total_tokens_sum for r in round_summaries if r.usage_total_tokens_sum
    ]

    return {
        "model_id": model_id,
        "num_games": num_games,
        "success_count": success_count,
        "success_rate": success_count / num_games if num_games else 0.0,
        "avg_success_steps": avg_success_steps,
        "avg_steps_overall": avg_steps_overall,
        "avg_round_time_sec": avg_round_time_sec,
        "avg_round_cot_tokens": avg_round_cot_tokens,
        "avg_round_total_tokens": avg_round_total_tokens,
        "avg_tokens_per_step": avg_tokens_per_step,
        "avg_step_infer_time_sec": avg_step_infer_time_sec,
        "token_count_fallback": token_count_fallback,
        "avg_usage_prompt_tokens": (
            sum(usage_prompt_tokens) / len(usage_prompt_tokens)
            if usage_prompt_tokens
            else None
        ),
        "avg_usage_completion_tokens": (
            sum(usage_completion_tokens) / len(usage_completion_tokens)
            if usage_completion_tokens
            else None
        ),
        "avg_usage_total_tokens": (
            sum(usage_total_tokens) / len(usage_total_tokens) if usage_total_tokens else None
        ),
        "total_steps_recorded": total_steps_recorded,
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_outputs(
    stats_root: str,
    timestamp: str,
    model_id: str,
    round_summaries: List[RoundSummary],
    step_records: List[Dict[str, Any]],
) -> Tuple[str, str, Optional[str]]:
    json_dir = os.path.join(stats_root, "json")
    metrics_dir = os.path.join(stats_root, "metrics")
    ensure_dir(json_dir)
    ensure_dir(metrics_dir)

    steps_path = os.path.join(json_dir, f"steps_{model_id}_{timestamp}.jsonl")
    rounds_path = os.path.join(json_dir, f"rounds_{model_id}_{timestamp}.jsonl")

    write_jsonl(steps_path, step_records)
    write_jsonl(rounds_path, [asdict(r) for r in round_summaries])

    # 仅在启用 COT 精炼时额外写出“精炼后仍与原动作一致”的摘要。
    refine_matches_path: Optional[str] = None
    try:
        matches = []
        for rec in step_records:
            refine = rec.get("cot_refine") or {}
            if not refine or not refine.get("match"):
                continue
            state = rec.get("state") or {}
            hunters = rec.get("hunter_positions") or [
                h.get("position") for h in state.get("hunters", [])
            ]
            prey = rec.get("prey_position") or (state.get("prey") or {}).get("position")
            matches.append(
                {
                    "model_id": model_id,
                    "round_id": rec.get("round_id"),
                    "step_id": rec.get("step_id"),
                    "hunter_positions": hunters,
                    "prey_position": prey,
                    "cot_text": rec.get("cot_text"),
                    "cot_tokens": rec.get("cot_tokens"),
                    "refined_cot_text": refine.get("refined_cot_text"),
                    "refined_cot_tokens": refine.get("refined_cot_tokens"),
                    "final_ratio": refine.get("final_ratio"),
                    "original_infer_time_sec": rec.get("step_infer_time_sec"),
                    "refine_infer_time_sec": refine.get("refine_total_infer_time_sec"),
                }
            )
        refine_matches_path = os.path.join(
            json_dir, f"cot_refine_matches_{model_id}_{timestamp}.jsonl"
        )
        write_jsonl(refine_matches_path, matches)
    except Exception as exc:
        print(f"[{model_id}] 写入精炼COT摘要失败: {exc}", flush=True)
        refine_matches_path = None

    return steps_path, rounds_path, refine_matches_path


def build_model_config(args: argparse.Namespace, model_id: str) -> Dict[str, Any]:
    config = MODEL_CONFIGS.get(model_id, {}).copy()
    if model_id == "qwen0.5b":
        config["tokenizer_path"] = args.qwen05_path
    elif model_id == "qwen7b":
        config["tokenizer_path"] = args.qwen7b_path

    if model_id == "deepseek":
        config["model_name"] = args.deepseek_model_name
        config["base_url"] = args.remote_base_url
    else:
        config["base_url"] = args.local_base_url
    config["backend"] = args.backend
    return config


def evaluate_for_model(
    model_id: str,
    args: argparse.Namespace,
    timestamp: str,
) -> Tuple[Dict[str, Any], str, str, Optional[str]]:
    config = build_model_config(args, model_id)
    token_counter = TokenCounter(model_id=model_id, tokenizer_path=config.get("tokenizer_path"))
    runner = LLMRunner(
        model_id=model_id,
        model_name=config["model_name"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=config["base_url"],
        api_key=args.api_key,
        backend=config["backend"],
        tokenizer_path=config.get("tokenizer_path"),
    )

    logs_dir = os.path.join(args.stats_dir, "logs")
    cot_log_path = os.path.join(logs_dir, f"cot_{model_id}_{timestamp}.log")
    progress_log_path = os.path.join(logs_dir, f"progress_{model_id}_{timestamp}.log")
    cot_log = _open_log_file(cot_log_path)
    progress_log = _open_log_file(progress_log_path)
    cot_logger = (lambda rec: log_cot_step(cot_log, rec)) if cot_log else None
    print(f"[{model_id}] COT日志: {cot_log_path}", flush=True)
    print(f"[{model_id}] 进度日志: {progress_log_path}", flush=True)

    round_summaries: List[RoundSummary] = []
    step_records: List[Dict[str, Any]] = []
    print(f"[{model_id}] 开始评估 {args.num_games} 轮...", flush=True)
    progress_every = max(1, args.num_games // 10)
    total_steps_so_far = 0
    fallback_steps_so_far = 0
    parse_ok_steps_so_far = 0
    round_bar = create_progress(
        args.num_games,
        f"[{model_id}] 轮次",
        enable=not args.no_progress,
        leave=True,
    )
    try:
        for round_idx in range(1, args.num_games + 1):
            if args.seed is not None:
                seed_everything(args.seed + round_idx * 97)
            game = HuntingGame(
                model_id=model_id,
                runner=runner,
                token_counter=token_counter,
                grid_size=args.grid_size,
                max_steps=args.max_steps,
                visualize=args.visualize,
                cot_refine_enabled=args.cot_refine,
                cot_refine_start=args.cot_refine_start,
                cot_refine_step=args.cot_refine_step,
                cot_refine_max=args.cot_refine_max,
                cot_refine_temperature=args.cot_refine_temperature,
                cot_refine_action_temperature=args.cot_refine_action_temperature,
                cot_refine_action_max_tokens=args.cot_refine_action_max_tokens,
            )
            round_summary, steps = game.run_round(
                round_idx,
                show_progress=not args.no_progress,
                cot_logger=cot_logger,
            )
            round_summaries.append(round_summary)
            step_records.extend(steps)
            total_steps_so_far += len(steps)
            fallback_steps_so_far += sum(1 for s in steps if s.get("fallback_used"))
            parse_ok_steps_so_far += sum(1 for s in steps if s.get("llm_parse_ok"))
            log_progress_line(
                progress_log,
                model_id=model_id,
                round_idx=round_idx,
                total_rounds=args.num_games,
                round_summary=round_summary,
                round_summaries=round_summaries,
                total_steps=total_steps_so_far,
                fallback_steps=fallback_steps_so_far,
                parse_ok_steps=parse_ok_steps_so_far,
            )
            if round_bar:
                round_bar.update(1)
                if hasattr(round_bar, "set_postfix_str"):
                    round_bar.set_postfix_str(
                        f"成功:{'Y' if round_summary.success else 'N'} 步数:{round_summary.steps_used}"
                    )
            elif (
                round_idx == args.num_games
                or round_idx % progress_every == 0
                or round_idx == 1
            ):
                print(f"[{model_id}] 进度 {round_idx}/{args.num_games}", flush=True)
    finally:
        if round_bar:
            round_bar.close()
        if cot_log:
            cot_log.close()
        if progress_log:
            progress_log.close()
    print(f"[{model_id}] 评估完成。", flush=True)

    steps_path, rounds_path, refine_matches_path = save_outputs(
        stats_root=args.stats_dir,
        timestamp=timestamp,
        model_id=model_id,
        round_summaries=round_summaries,
        step_records=step_records,
    )

    metrics = aggregate_model_metrics(model_id, round_summaries, step_records)
    return metrics, steps_path, rounds_path, refine_matches_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pursuit game with configurable LLMs.")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="deepseek")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Run multiple models sequentially.",
    )
    parser.add_argument("--num_games", type=int, default=50, help="Number of evaluation rounds.")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--visualize", action="store_true", help="Enable visualization/GIF.")
    parser.add_argument("--temperature", type=float, default=0.7)

    # COT精炼实验：仅对 deepseek(云端) 生效
    parser.add_argument("--cot_refine", action="store_true", help="Enable per-step COT refinement search for DeepSeek.")
    parser.add_argument("--cot_refine_start", type=float, default=0.10, help="Start ratio (0~1). e.g. 0.30 means refine COT to ~30% tokens.")
    parser.add_argument("--cot_refine_step", type=float, default=0.05, help="Increase ratio step when decisions mismatch.")
    parser.add_argument("--cot_refine_max", type=float, default=1.00, help="Max ratio (0~1).")
    parser.add_argument("--cot_refine_temperature", type=float, default=0.7, help="Temperature for refinement call.")
    parser.add_argument("--cot_refine_action_temperature", type=float, default=0.7, help="Temperature for action re-run with refined COT.")
    parser.add_argument("--cot_refine_action_max_tokens", type=int, default=128, help="Max tokens for refined-action JSON output.")
    # 对齐 glm_evaluate111.py：给足输出长度，避免思考过程被截断。
    # openai backend: max_tokens；transformers backend: max_new_tokens。
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_base_url", type=str, default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--remote_base_url", type=str, default=DEFAULT_REMOTE_BASE_URL)
    parser.add_argument("--deepseek_model_name", type=str, default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--api_key", type=str, default=None, help="Direct API key (overrides env).")
    parser.add_argument("--api_key_env", type=str, default="SILICONFLOW_API_KEY")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "transformers"])
    parser.add_argument("--qwen05_path", type=str, default=DEFAULT_QWEN05_PATH)
    parser.add_argument("--qwen7b_path", type=str, default=DEFAULT_QWEN7B_PATH)
    parser.add_argument("--stats_dir", type=str, default="统计目录")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file to append summary.")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="禁用进度条输出（默认开启）。",
    )
    return parser.parse_args()


def print_console_summary(
    all_metrics: List[Dict[str, Any]], output_files: List[str], log_file: Optional[str] = None
) -> None:
    lines: List[str] = []
    lines.append("\n" + "=" * 60)
    lines.append("评估结果汇总")
    lines.append("=" * 60)
    overall_rounds = sum(m["num_games"] for m in all_metrics)
    overall_successes = sum(m.get("success_count", 0) for m in all_metrics)
    overall_steps = sum(m.get("total_steps_recorded", 0) for m in all_metrics)
    overall = {
        "success_rate": (overall_successes / overall_rounds) if overall_rounds else 0.0,
        "avg_round_time_sec": (
            sum(m["avg_round_time_sec"] * m["num_games"] for m in all_metrics) / overall_rounds
            if overall_rounds
            else 0.0
        ),
        "avg_round_total_tokens": (
            sum(m["avg_round_total_tokens"] * m["num_games"] for m in all_metrics) / overall_rounds
            if overall_rounds
            else 0.0
        ),
        "avg_tokens_per_step": (
            sum(m.get("avg_tokens_per_step", 0.0) * m.get("total_steps_recorded", 0) for m in all_metrics)
            / overall_steps
            if overall_steps
            else 0.0
        ),
    }

    for m in all_metrics:
        lines.append(
            f"模型 {m['model_id']}: 轮次 {m['num_games']} | 成功率 {m['success_rate']*100:.1f}% "
            f"({m.get('success_count', 0)}/{m['num_games']}) | 平均成功步数 {m['avg_success_steps']:.1f} | "
            f"平均每轮耗时 {m['avg_round_time_sec']:.2f}s | 平均每轮token(cot/总) "
            f"{m['avg_round_cot_tokens']:.1f}/{m['avg_round_total_tokens']:.1f} | 平均每步token(总) "
            f"{m['avg_tokens_per_step']:.1f} | 平均每轮token(总) {m['avg_round_total_tokens']:.1f} | "
            f"平均每步推理耗时 {m['avg_step_infer_time_sec']:.3f}s"
        )

    lines.append("-" * 60)
    lines.append(
        f"Overall: 轮次 {overall_rounds} | 成功率 {overall['success_rate']*100:.1f}% "
        f"({overall_successes}/{overall_rounds}) | 平均每轮耗时 {overall['avg_round_time_sec']:.2f}s | "
        f"平均每步token(总) {overall['avg_tokens_per_step']:.1f} | "
        f"平均每轮token(总) {overall['avg_round_total_tokens']:.1f}"
    )
    lines.append("-" * 60)
    for path in output_files:
        lines.append(f"数据文件: {path}")
    lines.append("=" * 60 + "\n")

    for line in lines:
        print(line)

    if log_file:
        try:
            ensure_dir(os.path.dirname(log_file))
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"日志已写入: {log_file}")
        except Exception as exc:
            print(f"写入日志失败: {exc}")


def main():
    args = parse_args()
    models = args.models if args.models else [args.model]
    args.api_key = args.api_key or os.environ.get(args.api_key_env, "")
    requires_remote = "deepseek" in models
    if not args.api_key and requires_remote:
        print(f"警告: 环境变量 {args.api_key_env} 未设置，远端调用可能失败。")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics: List[Dict[str, Any]] = []
    output_files: List[str] = []

    for model_id in models:
        metrics, steps_path, rounds_path, refine_matches_path = evaluate_for_model(model_id, args, timestamp)
        all_metrics.append(metrics)
        output_files.extend([steps_path, rounds_path])
        if refine_matches_path:
            output_files.append(refine_matches_path)

    metrics_dir = os.path.join(args.stats_dir, "metrics")
    ensure_dir(metrics_dir)
    metrics_path = os.path.join(metrics_dir, f"metrics_summary_{timestamp}.json")
    payload = {
        "timestamp": timestamp,
        "params": {
            "num_games": args.num_games,
            "max_steps": args.max_steps,
            "grid_size": args.grid_size,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "local_base_url": args.local_base_url,
            "remote_base_url": args.remote_base_url,
            "backend": args.backend,
            "models": models,
            "seed": args.seed,
        },
        "models": {m["model_id"]: m for m in all_metrics},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    output_files.append(metrics_path)

    print_console_summary(all_metrics, output_files, log_file=args.log_file)


if __name__ == "__main__":
    main()
