#!/usr/bin/env python3
"""
CoT精炼脚本 - 完整版
包含迭代搜索最优精炼比例的核心功能
"""
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

API_KEY = "sk-rsjwrlxmxamwaulyhznvmxbogcgbsbxxcwhudpqpxqnlpeum"  # ⚠️ 你的 Key
BASE_URL = "https://api.siliconflow.cn/v1"
# 模型名称
CLOUD_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp"
# 注意：如果 V3.2 不可用，可尝试 "deepseek-ai/DeepSeek-V3" 或 "Qwen/Qwen2.5-72B-Instruct"

DATASET_PATH = "/home/liuheng/edge_cloude/统计目录/realtime_escort_nocollision/refine_data2.jsonl"  # ⚠️ 输入数据集路径

# 初始化客户端
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    print(f"✅ API 客户端初始化成功，使用模型: {CLOUD_MODEL_NAME}")
except Exception as e:
    print(f"❌ API 客户端初始化失败: {e}")
    sys.exit(1)


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


class SimpleProgress:
    """简单的进度条（无外部依赖）- 增强版，显示剩余时间"""
    def __init__(self, total: int, desc: str, leave: bool = False):
        self.total = max(1, total)
        self.desc = desc
        self.leave = leave
        self.current = 0
        self.last_percent = -1
        self._isatty = sys.stdout.isatty()
        self.start_time = time.time()
        if self._isatty:
            self._render(force=True)

    def _render(self, force: bool = False) -> None:
        if not self._isatty:
            return
        percent = int((self.current / self.total) * 100)
        
        # 计算剩余时间
        elapsed = time.time() - self.start_time
        if self.current > 0:
            avg_per_item = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = avg_per_item * remaining_items
            
            # 格式化剩余时间
            if eta_seconds < 60:
                eta_str = f"{int(eta_seconds)}s"
            elif eta_seconds < 3600:
                eta_str = f"{int(eta_seconds/60)}m{int(eta_seconds%60)}s"
            else:
                eta_str = f"{int(eta_seconds/3600)}h{int((eta_seconds%3600)/60)}m"
            
            time_info = f"ETA:{eta_str}"
        else:
            time_info = "计算中..."
        
        bar_len = 20
        filled = int(bar_len * self.current / self.total)
        bar = "#" * filled + "." * (bar_len - filled)
        
        # 显示更详细的信息
        print(
            f"{self.desc} [{bar}] {self.current}/{self.total} ({percent}%) {time_info}",
            end="\r",
            flush=True,
        )
        
        self.last_percent = percent

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        # 每次update都重新渲染，确保进度条实时更新
        self._render(force=True)

    def close(self) -> None:
        if self._isatty and not self.leave:
            print(" " * 80, end="\r")
        print(f"{self.desc} 完成 {self.current}/{self.total}", flush=True)


def create_progress(total: int, desc: str, enable: bool = True, leave: bool = False):
    """创建进度条"""
    if not enable:
        return None
    return SimpleProgress(total, desc, leave=leave)


class TokenCounter:
    """Token计数器（使用transformers或字节长度fallback）"""
    def __init__(self, tokenizer_path: Optional[str] = None):
        self.tokenizer = None
        self.backend = "len(text_bytes)"
        
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.backend = f"transformers:{tokenizer_path}"
            except Exception as exc:
                print(f"警告: transformers tokenizer加载失败: {exc}", file=sys.stderr)

    def count(self, text: str) -> int:
        """计算文本的token数"""
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return len(text.encode("utf-8"))


class LLMRunner:
    """LLM推理Runner（使用全局client）"""
    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def infer(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], float]:
        """调用LLM进行推理"""
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=CLOUD_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - start
        
        text = (response.choices[0].message.content or "").strip()
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }
        return text, usage, elapsed


def extract_actions_from_text(
    response_text: str,
) -> Tuple[Optional[List[int]], str, str, bool, bool, Optional[str]]:
    """从模型输出解析actions（完整保留原代码逻辑）"""

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
                "stay": 0, "停留": 0, "无动作": 0, "不动": 0,
                "up": 1, "上": 1,
                "down": 2, "下": 2,
                "left": 3, "左": 3,
                "right": 4, "右": 4,
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
        v0 = find_key(parsed, "guard0")
        v1 = find_key(parsed, "guard1")
        v2 = find_key(parsed, "guard2")
        if v0 is None or v1 is None or v2 is None:
            return None, "missing guard keys"
        try:
            return [coerce_action(v0), coerce_action(v1), coerce_action(v2)], None
        except Exception as exc:
            return None, str(exc)

    response_text = response_text or ""
    action_json_text = ""
    parse_ok = False
    fallback_used = False
    error: Optional[str] = None

    thought = response_text.strip()
    if "```" in response_text:
        thought = response_text.split("```")[0].strip()

    candidates: List[str] = []

    # 1) 解析代码块
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", response_text, re.S)
    for block in reversed(fenced_blocks):
        block = block.strip()
        if block:
            candidates.append(block)

    # 2) 提取平衡括号的JSON
    lowered = response_text.lower()
    key_pos = lowered.rfind("guard0")
    if key_pos != -1:
        start = response_text.rfind("{", 0, key_pos)
        if start != -1:
            extracted = extract_balanced_json(response_text, start)
            if extracted:
                candidates.append(extracted.strip())

    # 3) 直接就是JSON
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

    # 4) 兜底：从文本抽取键值对
    if actions is None:
        def _extract_guard_action(guard_idx: int) -> Optional[int]:
            m = re.search(rf"guard{guard_idx}\s*[:=]\s*([0-4])", response_text, re.I)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            m = re.search(rf"guard{guard_idx}\s*[:=]\s*(stay|up|down|left|right|停留|上|下|左|右)", response_text, re.I)
            if m:
                try:
                    return coerce_action(m.group(1))
                except Exception:
                    return None
            m = re.search(rf"(?:\*\*\s*)?Guard\s*{guard_idx}(?:\s*\*\*)?\s*[:：][^\n]*?(stay|up|down|left|right|停留|上|下|左|右)", response_text, re.I)
            if m:
                try:
                    return coerce_action(m.group(1))
                except Exception:
                    return None
            return None

        g0 = _extract_guard_action(0)
        g1 = _extract_guard_action(1)
        g2 = _extract_guard_action(2)
        if g0 is not None and g1 is not None and g2 is not None:
            actions = [g0, g1, g2]
            action_json_text = json.dumps({"guard0": actions[0], "guard1": actions[1], "guard2": actions[2]}, ensure_ascii=False)
            parse_ok = True
            error = None

    if actions is None:
        fallback_used = True

    return actions, thought, action_json_text, parse_ok, fallback_used, error


def build_prompt_from_state(state: Dict[str, Any], vip_next_pos: List[int], formation_limit: int = 3, grid_size: int = 20, max_steps: int = 100) -> str:
    """从state字段和VIP下一步位置构建game prompt"""
    step = state.get("step", 0)
    guards = state.get("guards", [])
    vip = state.get("vip", {})
    goal = state.get("goal", [0, 0])
    
    # 计算每个保镖到VIP下一步位置的曼哈顿距离
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    dists_to_next = [
        manhattan_distance(guards[0]["position"], vip_next_pos),
        manhattan_distance(guards[1]["position"], vip_next_pos),
        manhattan_distance(guards[2]["position"], vip_next_pos)
    ]

    prompt = f"""你正在控制三个保镖(Bodyguard 0, 1, 2)在20x20的网格中保护一名VIP(A)。

【坐标系定义】重要！
- 网格格式: (行 row, 列 col)
- 行 (Row) 范围: 0-19 (向下增加)
- 列 (Col) 范围: 0-19 (向右增加)
- 动作映射:
  0: 停留 (row, col)
  1: 上 (row-1, col)  <-- 减少行号
  2: 下 (row+1, col)  <-- 增加行号
  3: 左 (row, col-1)  <-- 减少列号
  4: 右 (row, col+1)  <-- 增加列号

【任务目标】
1. VIP(A) 正在前往终点 {goal}。
2. 你必须控制保镖，使它们与 **VIP的目标位置** 保持在 {formation_limit} 个曼哈顿距离以内。
3. **曼哈顿距离计算公式**: |r1 - r2| + |c1 - c2|
4. **忽略碰撞**: 单位可以重叠。

【当前状态】
- 步数: {step}/100
- ★★★ VIP即将移动到的位置 (Target): {list(vip_next_pos)} ★★★ 

【保镖详情】
Bodyguard 0: 当前 {guards[0]['position']} -> 距Target: {dists_to_next[0]}
Bodyguard 1: 当前 {guards[1]['position']} -> 距Target: {dists_to_next[1]}
Bodyguard 2: 当前 {guards[2]['position']} -> 距Target: {dists_to_next[2]}

【思考步骤要求】
对每个保镖，请执行以下步骤：
1. 计算 **相对位置向量**: (Target行 - 保镖行, Target列 - 保镖列)。
2. 判定 **必要移动**: 
   - 如果行差 > 0，说明Target在下方，应向下(2)；如果行差 < 0，应向上(1)。
   - 如果列差 > 0，说明Target在右方，应向右(4)；如果列差 < 0，应向左(3)。
3. **紧急检查**: 如果当前距离已经 >= {formation_limit - 1}，必须立刻朝 Target 方向移动，绝对不能"停留"！

【输出格式】
1.请你输出自己的思考过程，用有序列表，每条为一句可读的推理步骤与思考方式
2.思考过程不要有直接的动作结果，不要出现直接的决策过程，最后输出 JSON (用```包裹):
```{{
  "guard0": 动作编号,
  "guard1": 动作编号,
  "guard2": 动作编号
}}```
"""
    return prompt


def parse_actions_field(action_field: Any) -> Optional[List[int]]:
    """从数据集的actions字段解析ground-truth actions"""
    if action_field is None:
        return None
    # 数据集中的actions已经是列表格式，直接使用
    if isinstance(action_field, list) and len(action_field) == 3:
        try:
            return [int(action_field[0]), int(action_field[1]), int(action_field[2])]
        except Exception:
            return None
    # 兼容可能的字典格式（使用guard0/1/2）
    if isinstance(action_field, dict):
        try:
            return [int(action_field["guard0"]), int(action_field["guard1"]), int(action_field["guard2"])]
        except Exception:
            return None
    # 兼容字符串格式
    if isinstance(action_field, str):
        s = action_field.strip()
        parsed_actions, _thought, _aj, ok, _fb, _err = extract_actions_from_text(s)
        if ok and parsed_actions is not None:
            return parsed_actions
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and all(k in obj for k in ("guard0","guard1","guard2")):
                return [int(obj["guard0"]), int(obj["guard1"]), int(obj["guard2"])]
        except Exception:
            return None
    return None


class COTRefinementEngine:
    """CoT精炼引擎（完整版，包含迭代搜索）"""

    def __init__(
        self,
        model_id: str,
        runner: LLMRunner,
        token_counter: TokenCounter,
        grid_size: int = 20,
        max_steps: int = 100,
        cot_refine_start: float = 0.10,
        cot_refine_step: float = 0.05,
        cot_refine_max: float = 1.00,
        cot_refine_temperature: float = 0.5,
        cot_refine_action_temperature: float = 0.5,
        cot_refine_action_max_tokens: int = 128,
    ):
        self.model_id = model_id
        self.runner = runner
        self.token_counter = token_counter
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        self.cot_refine_start = float(cot_refine_start)
        self.cot_refine_step = float(cot_refine_step)
        self.cot_refine_max = float(cot_refine_max)
        self.cot_refine_temperature = float(cot_refine_temperature)
        self.cot_refine_action_temperature = float(cot_refine_action_temperature)
        self.cot_refine_action_max_tokens = int(cot_refine_action_max_tokens)

    def count_tokens(self, text: str) -> int:
        """计算token数"""
        return self.token_counter.count(text or "")

    def _build_system_message(self) -> Dict[str, str]:
        """构建system message (参考game2.py格式)"""
        return {
            "role": "system",
            "content": "你是一个多智能体协同护航专家。已知VIP的下一步位置，请控制保镖跟随。",
        }

    def _refine_cot_once(
        self,
        original_cot: str,
        ratio: float,
        original_cot_tokens: int,
    ) -> Tuple[str, Dict[str, int], float, int]:
        """精炼CoT到目标比例（删除最终决策）"""
        ratio = float(max(0.0, min(1.0, ratio)))
        target_tokens = max(32, int(original_cot_tokens * ratio))

        sys_msg = {
            "role": "system",
            "content": (
                '你是一个COT精炼器。你将收到一段包含推理与最终动作结论的思维链(COT)。'
                '请将其压缩/精炼到指定比例，但【必须删除或改写所有最终决策信息】：'
                '包括任何明确的动作选择/推荐、动作编号(0-4)、以及诸如"最终/因此/选择/决定/动作结果"等结论性语句。'
                '你只保留能让另一个模型在看到同一状态后独立推出动作所需的关键信息：'
                'VIP与保镖位置、距离/差值计算、各候选方向对距离的影响(例如每个方向距离变化)、边界/碰撞等约束、'
                '以及通用决策规则/偏好(用条件规则表达，而不是给出结论)。'
                '只输出精炼后的COT正文文本；不要输出JSON，不要输出解释，不要添加标题。'
            ),
        }
        
        user_msg = {
            "role": "user",
            "content": (
                f"压缩比例(保留token比例)：{ratio:.2%}\n"
                f"原始COT token数：{original_cot_tokens}\n"
                f"目标token数(约)：{target_tokens}\n\n"
                "要求：\n"
                "1) 只能输出精炼后的COT正文；不要输出任何其它文字。\n"
                "2) 尽量让精炼后的COT长度接近目标token数(允许±10%)。\n"
                "3) 必须保留让模型能够独立推出动作所需的信息：位置、距离/差值、候选方向的距离变化、边界/碰撞约束、通用决策规则/偏好。\n"
                "4) 严禁包含任何最终决策：不要出现动作编号0-4；不要出现 guard0/guard1/guard2 的最终动作；不要出现\"最终/因此/选择/决定/动作结果\"等结论句。\n"
                "   允许列出候选方向(上/下/左/右/停留)及其评估，但不要指出哪个被选中。\n\n"
                f"原始COT：\n{original_cot}\n"
            ),
        }

        max_tokens = min(self.runner.max_tokens, max(64, int(target_tokens * 2.0)))
        refined_text, usage, infer_time = self.runner.infer(
            messages=[sys_msg, user_msg],
            temperature=self.cot_refine_temperature,
            max_tokens=int(max_tokens),
        )

        refined_text = (refined_text or "").strip()
        # 🆕 使用API返回的真实token值（而非估算）
        refined_tokens = usage.get("completion_tokens", 0) if usage else 0

        return refined_text, usage, infer_time, refined_tokens

    def _infer_actions_with_refined_cot(
        self,
        prompt: str,
        refined_cot: str,
    ) -> Tuple[Optional[List[int]], Dict[str, Any], float, bool, Optional[str]]:
        """用精炼后的CoT生成action决策"""
        sys_msg = self._build_system_message()
        user1 = {
            "role": "user",
            "content": (
                prompt
                + "\n\n注意：接下来会提供一段'精炼推理材料'，其中不包含最终决策。"
                + "请你基于该材料自行完成推理并输出最终动作JSON。"
                + "如果材料中缺少某些数值，你可以根据上面的状态自行补算；但不要假设任何未给出的'已选动作'结论。"
            ),
        }
        assistant_thought = {"role": "assistant", "content": refined_cot}
        user2 = {
            "role": "user",
            "content": (
                "请严格只输出最终动作 JSON（对象格式，键为 guard0/guard1/guard2，值为0-4）。"
                "不要输出任何解释或多余文本。"
            ),
        }

        text, usage, infer_time = self.runner.infer(
            [sys_msg, user1, assistant_thought, user2],
            temperature=self.cot_refine_action_temperature,
            max_tokens=min(self.runner.max_tokens, int(self.cot_refine_action_max_tokens)),
        )
        
        actions, _thought, _action_json_text, parse_ok, _fallback_used, parse_error = extract_actions_from_text(text)
        return actions, usage or {}, infer_time, bool(parse_ok), parse_error

    def search_matching_refined_cot(
        self,
        prompt: str,
        original_cot: str,
        target_actions: List[int],
        original_cot_api_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        核心迭代搜索逻辑：
        从start_ratio开始，每次精炼CoT，然后用精炼的CoT生成action。
        如果action与target_actions一致，返回成功；否则增加ratio继续。
        
        Args:
            original_cot_api_tokens: API返回的真实token值（优先使用）
        """
        original_cot = (original_cot or "").strip()
        
        # 优先使用API返回的真实token值
        if original_cot_api_tokens is not None:
            original_cot_tokens = original_cot_api_tokens
            token_source = "api"
        else:
            original_cot_tokens = self.count_tokens(original_cot)
            token_source = "estimated"

        start_ratio = max(0.0, min(1.0, self.cot_refine_start))
        step = max(1e-6, self.cot_refine_step)
        max_ratio = max(start_ratio, min(1.0, self.cot_refine_max))

        attempts = 0
        total_refine_time = 0.0
        total_action_time = 0.0
        total_refine_calls = 0
        total_action_calls = 0
        last_refine_usage: Dict[str, int] = {}
        last_action_usage: Dict[str, int] = {}

        last_refined: Optional[str] = None
        last_refined_tokens: Optional[int] = None
        last_ratio: Optional[float] = None
        last_actions: Optional[List[int]] = None

        ratio = start_ratio
        max_iters = int(math.ceil((max_ratio - start_ratio) / step)) + 2

        for _ in range(max_iters):
            if ratio > max_ratio + 1e-12:
                break

            attempts += 1

            # 精炼CoT
            refined_text, refine_usage, refine_time, refined_tokens = self._refine_cot_once(
                original_cot=original_cot,
                ratio=ratio,
                original_cot_tokens=original_cot_tokens,
            )
            total_refine_time += float(refine_time or 0.0)
            total_refine_calls += 1
            last_refine_usage = refine_usage or {}

            last_refined = refined_text
            last_refined_tokens = int(refined_tokens or 0)
            last_ratio = float(ratio)

            # 用精炼的CoT生成action
            actions, action_usage, action_time, action_parse_ok, action_parse_error = self._infer_actions_with_refined_cot(prompt, refined_text)
            total_action_time += float(action_time or 0.0)
            total_action_calls += 1
            last_action_usage = action_usage or {}
            last_actions = actions

            # 检查是否匹配
            if actions == target_actions:
                return {
                    "refine_start_ratio": start_ratio,
                    "refine_step": step,
                    "refine_max_ratio": max_ratio,
                    "original_cot_tokens": original_cot_tokens,
                    "original_cot_tokens_source": token_source,
                    "final_ratio": float(ratio),
                    "match": True,
                    "refined_cot_text": refined_text,
                    "refined_cot_tokens": int(refined_tokens or 0),
                    "refined_actions": actions,
                    "attempts": attempts,
                    "refine_total_infer_time_sec": float(total_refine_time + total_action_time),
                    "refine_total_refine_calls": int(total_refine_calls),
                    "refine_total_action_calls": int(total_action_calls),
                    "refine_last_refine_usage": last_refine_usage,
                    "refine_last_action_usage": last_action_usage,
                }

            # 增加ratio
            ratio = round(ratio + step, 10)
            if ratio > 1.0:
                ratio = 1.0

        # 达到max_ratio仍未匹配
        return {
            "refine_start_ratio": start_ratio,
            "refine_step": step,
            "refine_max_ratio": max_ratio,
            "original_cot_tokens": original_cot_tokens,
            "original_cot_tokens_source": token_source,
            "final_ratio": None,  # 失败时为None
            "match": False,
            "refined_cot_text": last_refined,
            "refined_cot_tokens": last_refined_tokens,
            "refined_actions": last_actions,
            "attempts": attempts,
            "refine_total_infer_time_sec": float(total_refine_time + total_action_time),
            "refine_total_refine_calls": int(total_refine_calls),
            "refine_total_action_calls": int(total_action_calls),
            "refine_last_refine_usage": last_refine_usage,
            "refine_last_action_usage": last_action_usage,
            "last_ratio_tried": last_ratio,
        }


def process_dataset(
    engine: COTRefinementEngine,
    dataset_path: str,
    output_path: str,
    limit: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """处理数据集，对每条记录进行迭代搜索"""
    
    # 读取数据集
    print(f"读取数据集: {dataset_path}", flush=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    if limit:
        records = records[:limit]
    
    print(f"总记录数: {len(records)}", flush=True)
    
    # 准备输出文件（实时写入）
    ensure_dir(os.path.dirname(output_path))
    output_file = open(output_path, "w", encoding="utf-8")
    
    # 处理记录
    progress = create_progress(len(records), f"[{engine.model_id}] CoT精炼", enable=show_progress, leave=True) if show_progress else None
    
    total_seen = 0
    used_records = 0
    skipped_no_action = 0
    refined_ok = 0
    refine_attempts_sum = 0
    
    for idx, record in enumerate(records):
        total_seen += 1
        
        try:
            state = record.get("state", {})
            # escort数据集使用cot字段，而非cot0
            cot0 = record.get("cot", record.get("cot0", ""))
            # escort数据集使用actions字段
            action_field = record.get("actions", record.get("action"))
            
            gt_actions = parse_actions_field(action_field)
            cot0_actions, *_ = extract_actions_from_text(cot0)
            
            if gt_actions is None:
                skipped_no_action += 1
                row = {
                    "state": state,
                    "skip_reason": "action field parse failed (missing/invalid ground truth)",
                    "cot0_actions": cot0_actions,
                    "action_field_actions": None,
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                if progress:
                    progress.update(1)
                continue
            
            used_records += 1
            # 从记录中获取vip_next_pos，如果没有则使用VIP当前位置作为默认值
            vip_next_pos = record.get("vip_next_pos", state.get("vip", {}).get("position", [0, 0]))
            
            # 🆕 获取API返回的真实token值（来自game2.py的completion_tokens）
            api_tokens = record.get("tokens", None)
            
            prompt = build_prompt_from_state(
                state, 
                vip_next_pos=vip_next_pos,
                formation_limit=3,
                grid_size=engine.grid_size, 
                max_steps=engine.max_steps
            )
            
            # 核心：迭代搜索匹配的精炼CoT，传入API真实token值
            refine_result = engine.search_matching_refined_cot(
                prompt=prompt,
                original_cot=cot0,
                target_actions=gt_actions,
                original_cot_api_tokens=api_tokens,
            )
            
            if refine_result.get("match"):
                refined_ok += 1
                refine_attempts_sum += int(refine_result.get("attempts") or 0)
            
            row = {
                "state": state,
                "cot0_actions": cot0_actions,
                "action_field_actions": gt_actions,
                "refine": refine_result,
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            
        except Exception as exc:
            print(f"错误: 处理记录 {idx} 时出错: {exc}", flush=True)
            row = {"error": str(exc), "idx": idx}
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
        
        if progress:
            progress.update(1)
    
    if progress:
        progress.close()
    
    output_file.close()
    
    summary = {
        "model_id": engine.model_id,
        "dataset_path": dataset_path,
        "total_records_seen": total_seen,
        "used_records": used_records,
        "skipped_no_action_field": skipped_no_action,
        "refined_ok": refined_ok,
        "refined_ok_rate": (refined_ok / used_records) if used_records else 0.0,
        "avg_attempts_when_ok": (refine_attempts_sum / refined_ok) if refined_ok else None,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="CoT精炼脚本（完整版-迭代搜索）")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="输入数据集路径(jsonl)")
    parser.add_argument("--output_dir", type=str, default="dataset", help="输出目录")
    parser.add_argument("--limit", type=int, default=None, help="处理记录数上限")
    
    # 迭代搜索参数
    parser.add_argument("--cot_refine_start", type=float, default=0.10, help="起始精炼比例(0-1)")
    parser.add_argument("--cot_refine_step", type=float, default=0.05, help="每次增加的步长")
    parser.add_argument("--cot_refine_max", type=float, default=1.00, help="最大精炼比例(0-1)")
    
    # 温度参数
    parser.add_argument("--cot_refine_temperature", type=float, default=0.5, help="精炼CoT的temperature")
    parser.add_argument("--cot_refine_action_temperature", type=float, default=0.5, help="生成action的temperature")
    parser.add_argument("--cot_refine_action_max_tokens", type=int, default=128, help="action生成的max_tokens")
    
    parser.add_argument("--max_tokens", type=int, default=2000, help="CoT精炼的最大token数")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    args = parser.parse_args()
    
    
    # 初始化组件
    token_counter = TokenCounter()
    runner = LLMRunner(
        temperature=args.cot_refine_temperature,
        max_tokens=args.max_tokens,
    )
    engine = COTRefinementEngine(
        model_id="deepseek",
        runner=runner,
        token_counter=token_counter,
        cot_refine_start=args.cot_refine_start,
        cot_refine_step=args.cot_refine_step,
        cot_refine_max=args.cot_refine_max,
        cot_refine_temperature=args.cot_refine_temperature,
        cot_refine_action_temperature=args.cot_refine_action_temperature,
        cot_refine_action_max_tokens=args.cot_refine_action_max_tokens,
    )
    
    # 生成输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_basename = os.path.splitext(os.path.basename(args.dataset_path))[0]
    output_path = os.path.join(
        args.output_dir,
        f"dataset_cot0_refine_target_action_deepseek_{timestamp}.jsonl"
    )
    
    # 处理数据集
    print(f"开始CoT迭代精炼搜索...", flush=True)
    print(f"搜索范围: {args.cot_refine_start:.2%} -> {args.cot_refine_max:.2%}, 步长: {args.cot_refine_step:.2%}", flush=True)
    print(f"输出路径: {output_path}", flush=True)
    
    summary = process_dataset(
        engine=engine,
        dataset_path=args.dataset_path,
        output_path=output_path,
        limit=args.limit,
        show_progress=not args.no_progress,
    )
    
    # 打印总结
    print(f"\n{'='*60}", flush=True)
    print(f"处理完成!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"模型: {summary['model_id']}", flush=True)
    print(f"总记录数: {summary['total_records_seen']}", flush=True)
    print(f"使用记录: {summary['used_records']}", flush=True)
    print(f"跳过记录(无action): {summary['skipped_no_action_field']}", flush=True)
    print(f"精炼成功: {summary['refined_ok']}", flush=True)
    print(f"成功率: {summary['refined_ok_rate']*100:.1f}%", flush=True)
    if summary['avg_attempts_when_ok']:
        print(f"平均尝试次数(成功时): {summary['avg_attempts_when_ok']:.1f}", flush=True)
    print(f"输出文件: {output_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
