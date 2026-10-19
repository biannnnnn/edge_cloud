import random
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import time
import re  # 新增：用于正则解析坐标
from openai import OpenAI

# ==========================================
# ⚠️ 配置云端模型 API
# ==========================================
API_KEY = "sk-rsjwrlxmxamwaulyhznvmxbogcgbsbxxcwhudpqpxqnlpeum" # ⚠️ 你的 Key
BASE_URL = "https://api.siliconflow.cn/v1"

# 模型名称
CLOUD_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2-Exp" 
# 注意：如果 V3.2 不可用，可尝试 "deepseek-ai/DeepSeek-V3" 或 "Qwen/Qwen2.5-72B-Instruct"

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

# 超参数配置
GRID_SIZE = 20
MAX_FORMATION_DIST = 3  # BCD 必须保持在 A 的这个距离内
MIN_TRAVEL_DIST = 10    # 起点终点最小间隔
COORD_FILE_PATH = "vip_coordinates_2.txt" # 坐标文件路径

class EscortGame:
    def __init__(self, game_id=1, jsonl_file=None, custom_scenario=None):
        self.grid_size = GRID_SIZE
        self.max_steps = 100
        self.current_step = 0
        self.formation_limit = MAX_FORMATION_DIST

        # 初始化游戏场景
        # 如果传入了 custom_scenario (tuple: (start, end))，则使用它
        self.vip_pos, self.goal_pos, self.guards = self.setup_scenario(custom_scenario)
        
        self.action_map = {
            0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)
        }
        
        # 记录历史
        self.history = {
            'guard_0': [], 'guard_1': [], 'guard_2': [], 'vip': []
        }
        for i in range(3):
            self.history[f'guard_{i}'].append({
                'step': 0, 'position': self.guards[i], 'action': None
            })
        self.history['vip'].append({
            'step': 0, 'position': self.vip_pos, 'action': None
        })

        self.fig = None
        self.ax = None
        self.guard_scat = None
        self.vip_scat = None
        self.goal_scat = None
        self.frames = []
        
        self.total_output_tokens = 0
        self.game_id = game_id
        self.jsonl_file = jsonl_file

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def setup_scenario(self, custom_scenario=None):
        """
        设置场景：
        1. 确定 VIP 起点和终点 (从文件读取 或 随机生成)
        2. 基于 VIP 起点随机生成 3 个保镖位置
        """
        if custom_scenario:
            vip_pos = custom_scenario[0]
            goal_pos = custom_scenario[1]
            # 简单的校验
            if not (0 <= vip_pos[0] < self.grid_size and 0 <= vip_pos[1] < self.grid_size):
                print("⚠️ 警告：文件中的坐标超出网格范围，正在修正...")
                vip_pos = (max(0, min(19, vip_pos[0])), max(0, min(19, vip_pos[1])))
        else:
            # 随机生成 (保留原有逻辑作为兜底)
            while True:
                start = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
                end = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
                if self.manhattan_distance(start, end) > MIN_TRAVEL_DIST:
                    vip_pos = start
                    goal_pos = end
                    break
        
        # 生成保镖位置 (无论 VIP 位置怎么来的，保镖都在 VIP 身边随机生成)
        guards = []
        attempts = 0
        while len(guards) < 3 and attempts < 1000:
            dx = random.randint(-self.formation_limit, self.formation_limit)
            dy = random.randint(-self.formation_limit, self.formation_limit)
            gx = vip_pos[0] + dx
            gy = vip_pos[1] + dy
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                pos = (gx, gy)
                # 初始化时尽量还是分开，虽然移动时忽略碰撞
                if self.manhattan_distance(pos, vip_pos) <= self.formation_limit:
                    if pos not in guards and pos != vip_pos:
                        guards.append(pos)
            attempts += 1
            
        if len(guards) < 3:
            guards = [
                (max(0, min(19, vip_pos[0]+1)), vip_pos[1]),
                (max(0, min(19, vip_pos[0]-1)), vip_pos[1]),
                (vip_pos[0], max(0, min(19, vip_pos[1]+1)))
            ]
            
        return vip_pos, goal_pos, guards

    def check_formation_broken(self):
        distances = [self.manhattan_distance(g, self.vip_pos) for g in self.guards]
        return any(d > self.formation_limit for d in distances), distances

    def move_agent(self, position, action, occupied_positions=None):
        delta = self.action_map[action]
        new_pos = (position[0] + delta[0], position[1] + delta[1])
        new_pos = (
            max(0, min(self.grid_size - 1, new_pos[0])),
            max(0, min(self.grid_size - 1, new_pos[1]))
        )
        if occupied_positions is not None and new_pos in occupied_positions:
            return position
        return new_pos
    
    def get_vip_next_move(self):
        curr = self.vip_pos
        goal = self.goal_pos
        best_action = 0
        min_dist = self.manhattan_distance(curr, goal)
        actions = [1, 2, 3, 4]
        random.shuffle(actions)
        
        for action in actions:
            delta = self.action_map[action]
            next_pos = (curr[0] + delta[0], curr[1] + delta[1])
            if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                continue
            dist = self.manhattan_distance(next_pos, goal)
            if dist < min_dist:
                min_dist = dist
                best_action = action
                break 
        return best_action

    def get_state_description(self):
        dist_flag, distances = self.check_formation_broken()
        state = {
            "step": self.current_step,
            "guards": [{"id": i, "position": list(self.guards[i])} for i in range(3)],
            "vip": {"position": list(self.vip_pos)},
            "goal": list(self.goal_pos),
            "distances_to_vip": distances
        }
        return state

    def get_llm_actions(self, vip_next_pos):
        state = self.get_state_description()
        
        dists_to_next = [
            self.manhattan_distance(self.guards[0], vip_next_pos),
            self.manhattan_distance(self.guards[1], vip_next_pos),
            self.manhattan_distance(self.guards[2], vip_next_pos)
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
1. VIP(A) 正在前往终点 {state['goal']}。
2. 你必须控制保镖，使它们与 **VIP的目标位置** 保持在 {self.formation_limit} 个曼哈顿距离以内。
3. **曼哈顿距离计算公式**: |r1 - r2| + |c1 - c2|
4. **忽略碰撞**: 单位可以重叠。

【当前状态】
- 步数: {state['step']}/100
- ★★★ VIP即将移动到的位置 (Target): {list(vip_next_pos)} ★★★ 

【保镖详情】
Bodyguard 0: 当前 {state['guards'][0]['position']} -> 距Target: {dists_to_next[0]}
Bodyguard 1: 当前 {state['guards'][1]['position']} -> 距Target: {dists_to_next[1]}
Bodyguard 2: 当前 {state['guards'][2]['position']} -> 距Target: {dists_to_next[2]}

【思考步骤要求】
对每个保镖，请执行以下步骤：
1. 计算 **相对位置向量**: (Target行 - 保镖行, Target列 - 保镖列)。
2. 判定 **必要移动**: 
   - 如果行差 > 0，说明Target在下方，应向下(2)；如果行差 < 0，应向上(1)。
   - 如果列差 > 0，说明Target在右方，应向右(4)；如果列差 < 0，应向左(3)。
3. **紧急检查**: 如果当前距离已经 >= {self.formation_limit - 1}，必须立刻朝 Target 方向移动，绝对不能"停留"！

【输出格式】
1.请你输出自己的思考过程，用有序列表，每条为一句可读的推理步骤与思考方式
2.思考过程不要有直接的动作结果，不要出现直接的决策过程，最后输出 JSON (用```包裹):
```{{
  "guard0": 动作编号,
  "guard1": 动作编号,
  "guard2": 动作编号
}}```
"""

        try:
            inference_start_time = time.time()
            
            messages = [
                {"role": "system", "content": "你是一个多智能体协同护航专家。已知VIP的下一步位置，请控制保镖跟随。"},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model=CLOUD_MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9
            )

            inference_end_time = time.time()
            thinking_time = inference_end_time - inference_start_time
            
            response_text = response.choices[0].message.content
            
            if response.usage:
                num_new_tokens = response.usage.completion_tokens
            else:
                num_new_tokens = len(response_text)

            self.total_output_tokens += num_new_tokens
            
            print(f"☁️ API 响应耗时: {thinking_time:.2f}s | Tokens: {num_new_tokens}")

            original_response = response_text
            actions_inner = None
            
            if "```json" in original_response:
                actions_inner = original_response.split("```json")[1].split("```")[0].strip()
            elif "```" in original_response:
                actions_inner = original_response.split("```")[1].split("```")[0].strip()

            if "```" in original_response:
                thought = original_response.split("```")[0].strip()
            else:
                thought = original_response

            actions = [0, 0, 0]
            if actions_inner is not None:
                try:
                    actions_inner = actions_inner.replace("'", '"')
                    parsed = json.loads(actions_inner)
                    actions = [
                        parsed.get("guard0", 0),
                        parsed.get("guard1", 0),
                        parsed.get("guard2", 0)
                    ]
                except Exception:
                    actions = self.get_greedy_actions(vip_next_pos)
            else:
                actions = self.get_greedy_actions(vip_next_pos)

            actions = [int(max(0, min(4, int(a)))) for a in actions]

            if self.jsonl_file is not None:
                record = {
                    'game_id': self.game_id,
                    'step': self.current_step,
                    'state': state,
                    'vip_next_pos': list(vip_next_pos), 
                    'cot': thought,
                    'decision': json.dumps({"guard0": actions[0], "guard1": actions[1], "guard2": actions[2]}),
                    'actions': actions,
                    'tokens': num_new_tokens,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                self.jsonl_file.flush()
            
            return actions

        except Exception as e:
            print(f"❌ API 调用失败: {e}，使用默认策略")
            return self.get_greedy_actions(vip_next_pos)

    def get_greedy_actions(self, target_pos=None):
        if target_pos is None:
            target_pos = self.vip_pos
            
        actions = []
        for guard in self.guards:
            best_action = 0
            min_dist = float('inf')
            
            for action in range(5):
                new_pos = self.move_agent(guard, action, occupied_positions=None)
                dist = self.manhattan_distance(new_pos, target_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_action = action
            actions.append(best_action)
        return actions

    def step(self):
        self.current_step += 1
        
        vip_action = self.get_vip_next_move()
        vip_next_pos = self.move_agent(self.vip_pos, vip_action, occupied_positions=None) 
        
        guard_actions = self.get_llm_actions(vip_next_pos)
        
        new_guards = []
        for i in range(3):
            new_pos = self.move_agent(self.guards[i], guard_actions[i], occupied_positions=None)
            new_guards.append(new_pos)
        self.guards = new_guards
        
        for i in range(3):
            self.history[f'guard_{i}'].append({
                'step': self.current_step, 'position': self.guards[i], 'action': guard_actions[i]
            })

        self.vip_pos = vip_next_pos 
        self.history['vip'].append({
            'step': self.current_step, 'position': self.vip_pos, 'action': vip_action
        })
        
        return guard_actions, vip_action

    def render(self, mode='human', save_frame=True):
        if self.fig is None:
            import matplotlib
            matplotlib.use('Agg')
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
            self.ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
            self.ax.grid(True, color='gray', linewidth=0.5, alpha=0.6)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            self.ax.set_title("Escort Mission")
            
            guard_positions = np.array([self.guards[i] for i in range(3)])
            vip_position = np.array([self.vip_pos])
            goal_position = np.array([self.goal_pos])

            self.goal_scat = self.ax.scatter(
                goal_position[:, 1] + 0.5, goal_position[:, 0] + 0.5,
                c='green', s=400, marker='s', label='Goal', alpha=0.5
            )

            self.guard_scat = self.ax.scatter(
                guard_positions[:, 1] + 0.5, guard_positions[:, 0] + 0.5,
                c='blue', s=200, label='Bodyguards', marker='o', edgecolors='black', linewidths=2, alpha=0.7
            )
            
            self.vip_scat = self.ax.scatter(
                vip_position[:, 1] + 0.5, vip_position[:, 0] + 0.5,
                c='red', s=300, marker='*', label='VIP', edgecolors='black', linewidths=2
            )
            self.ax.legend(loc='upper right', fontsize=12)
        else:
            guard_positions = np.array([self.guards[i] for i in range(3)])
            vip_position = np.array([self.vip_pos])
            self.guard_scat.set_offsets(np.c_[guard_positions[:, 1] + 0.5, guard_positions[:, 0] + 0.5])
            self.vip_scat.set_offsets(np.c_[vip_position[:, 1] + 0.5, vip_position[:, 0] + 0.5])

        broken, dists = self.check_formation_broken()
        status_text = "BROKEN!" if broken else "OK"
        color = "red" if broken else "black"
        title = f"Step: {self.current_step} | Dists: {dists} | Status: {status_text}"
        self.ax.set_title(title, fontsize=14, color=color)

        if save_frame:
            self.fig.canvas.draw()
            import io
            from PIL import Image
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frame = np.array(img)[:, :, :3]
            buf.close()
            self.frames.append(frame.copy())

    def save_gif(self, result_status):
        if not self.frames: return
        gif_folder = "gif_escort_nocollision"
        if not os.path.exists(gif_folder): os.makedirs(gif_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_str = "WIN" if result_status == "win" else "FAIL"
        filename = os.path.join(gif_folder, f"{status_str}_{timestamp}.gif")
        try:
            from PIL import Image
            pil_images = [Image.fromarray(f if f.dtype==np.uint8 else (f*255).astype(np.uint8), mode='RGB') for f in self.frames]
            pil_images[0].save(filename, save_all=True, append_images=pil_images[1:], duration=100, loop=0)
            print(f"✅ GIF保存成功: {filename}")
        except Exception as e:
            print(f"❌ 保存GIF失败: {e}")

    def run(self, visualize=True):
        print("=" * 50)
        print(f"开始云端模型护送任务 (ID: {self.game_id})")
        print(f"VIP从 {self.vip_pos} 前往 {self.goal_pos}")
        if visualize: self.render()
        
        while self.current_step < self.max_steps:
            self.step()
            broken, dists = self.check_formation_broken()
            at_goal = (self.vip_pos == self.goal_pos)
            
            if visualize: self.render()
            
            if broken:
                print(f"❌ 任务失败！阵型破裂，距离: {dists}")
                if visualize: 
                    self.save_gif("fail")
                    plt.close(self.fig)
                return False
            
            if at_goal:
                print(f"🎉 任务成功！VIP安全抵达终点。")
                if visualize: 
                    self.save_gif("win")
                    plt.close(self.fig)
                return True
                
        print(f"❌ 任务失败！超过 {self.max_steps} 步")
        if visualize: 
            self.save_gif("timeout")
            plt.close(self.fig)
        return False

# ==========================================
# 辅助函数：读取坐标文件
# ==========================================
def load_vip_coordinates(filepath):
    """
    读取 txt 文件，每行格式: (x1, y1) (x2, y2)
    返回列表: [((r1, c1), (r2, c2)), ...]
    """
    coordinates = []
    pattern = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
    
    if not os.path.exists(filepath):
        print(f"⚠️ 文件不存在: {filepath}")
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                matches = pattern.findall(line)
                if len(matches) == 2:
                    start_pos = (int(matches[0][0]), int(matches[0][1]))
                    end_pos = (int(matches[1][0]), int(matches[1][1]))
                    coordinates.append((start_pos, end_pos))
                else:
                    print(f"⚠️ 第 {line_num} 行格式错误，已跳过: {line}")
        print(f"✅ 成功加载 {len(coordinates)} 组坐标。")
        return coordinates
    except Exception as e:
        print(f"❌ 读取坐标文件失败: {e}")
        return []

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    game_stats = []
    
    # 日志目录
    realtime_dir = os.path.join("统计目录", "realtime_escort_nocollision")
    os.makedirs(realtime_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    realtime_jsonl_path = os.path.join(realtime_dir, f"escort_log_{timestamp_str}.jsonl")
    
    # 统计目录2 (用于保存最终摘要)
    summary_dir = os.path.join("统计目录2")
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"📝 实时日志将保存至: {realtime_jsonl_path}\n")

    try:
        realtime_file = open(realtime_jsonl_path, 'w', encoding='utf-8')
        
        # 1. 加载坐标文件
        all_scenarios = load_vip_coordinates(COORD_FILE_PATH)
        
        if not all_scenarios:
            print("⚠️ 未加载到有效坐标，将使用随机生成模式。")
            max_available = 999
        else:
            max_available = len(all_scenarios)
        
        # 2. 获取用户输入
        input_str = input(f"请输入要进行的游戏轮次 (默认{min(5, max_available)}, 最大{max_available}): ").strip()
        num_games = int(input_str) if input_str else min(5, max_available)
        
        # 如果文件里的不够，就限制为文件数量
        if all_scenarios and num_games > max_available:
            print(f"⚠️ 请求的轮次超过文件行数，自动调整为 {max_available}")
            num_games = max_available

        vis_input = input("是否开启可视化 (y/n, 默认n): ").strip().lower()
        enable_vis = (vis_input == 'y')

        # 3. 循环运行游戏
        for i in range(num_games):
            print(f"\n🚀 启动第 {i+1}/{num_games} 轮护送...")
            start_time = time.time()
            
            # 提取当前轮次的固定坐标 (如果有)
            current_scenario = all_scenarios[i] if i < len(all_scenarios) else None
            
            game = EscortGame(game_id=i+1, jsonl_file=realtime_file, custom_scenario=current_scenario)
            result = game.run(visualize=enable_vis)
            duration = time.time() - start_time
            
            game_stats.append({
                "id": i + 1, 
                "success": result, 
                "steps": game.current_step,
                "duration": duration, 
                "tokens": game.total_output_tokens
            })

    except KeyboardInterrupt:
        print("\n⚠️  程序中断")
        
    finally:
        if 'realtime_file' in locals() and realtime_file:
            realtime_file.close()
            
        # 4. 生成并保存统计报告 (到统计目录2)
        if game_stats:
            total_games = len(game_stats)
            success_count = sum(1 for g in game_stats if g['success'])
            success_rate = (success_count / total_games) * 100
            
            avg_duration = sum(g['duration'] for g in game_stats) / total_games
            avg_tokens = sum(g['tokens'] for g in game_stats) / total_games
            avg_steps = sum(g['steps'] for g in game_stats) / total_games
            
            summary_content = [
                "========================================",
                "          多智能体护送任务统计报告",
                "========================================",
                f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"使用模型: {CLOUD_MODEL_NAME}",
                f"坐标来源: {COORD_FILE_PATH} ({'随机' if not all_scenarios else '文件'})",
                "----------------------------------------",
                f"运行总局数: {total_games}",
                f"成功率: {success_count}/{total_games} ({success_rate:.2f}%)",
                "----------------------------------------",
                "【平均指标】",
                f"平均每局耗时: {avg_duration:.2f} 秒",
                f"平均消耗Tokens: {avg_tokens:.2f}",
                f"平均步数: {avg_steps:.2f}",
                "----------------------------------------",
                "【每局明细】"
            ]
            
            for g in game_stats:
                status = "WIN " if g['success'] else "FAIL"
                summary_content.append(
                    f"ID:{g['id']:02d} | {status} | 步数:{g['steps']:3d} | 耗时:{g['duration']:6.2f}s | Tokens:{g['tokens']}"
                )
            
            # 写入文件
            summary_filename = f"statistics_summary_{timestamp_str}.txt"
            summary_path = os.path.join(summary_dir, summary_filename)
            
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(summary_content))
                print("\n" + "="*50)
                print(f"📊 最终统计报告已保存至: {summary_path}")
                print(f"   成功率: {success_rate:.2f}% | 平均Token: {avg_tokens:.0f}")
                print("="*50)
            except Exception as e:
                print(f"❌ 保存统计报告失败: {e}")