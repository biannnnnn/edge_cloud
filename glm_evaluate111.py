import random
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from openai import OpenAI
from datetime import datetime
import os
import sys
import logging
from tqdm import tqdm

# 初始化API客户端
client = OpenAI(
    api_key="sk-rsjwrlxmxamwaulyhznvmxbogcgbsbxxcwhudpqpxqnlpeum",  # ⚠️ 请在此处填入你的真实 API Key
    base_url="https://api.siliconflow.cn/v1"
)

class HuntingGame:
    def __init__(self, game_index=0):
        self.grid_size = 20
        self.max_steps = 100
        self.current_step = 0

        # 从文件读取初始化位置
        positions = self.load_coordinates_from_file(game_index)
        self.hunters = positions[:3]  # 三个猎人
        self.prey = positions[3]      # 一个猎物

        self.action_map = {
            0: (0, 0),   # 无动作
            1: (-1, 0),  # 上
            2: (1, 0),   # 下
            3: (0, -1),  # 左
            4: (0, 1)    # 右
        }

        self.action_names = ["停留", "上", "下", "左", "右"]

        # 记录历史路径
        self.history = {
            'hunter_0': [],
            'hunter_1': [],
            'hunter_2': [],
            'prey': []
        }

        # 记录初始位置
        for i in range(3):
            self.history[f'hunter_{i}'].append({
                'step': 0,
                'position': self.hunters[i],
                'action': None
            })
        self.history['prey'].append({
            'step': 0,
            'position': self.prey,
            'action': None
        })

        # 可视化相关
        self.fig = None
        self.ax = None
        self.hunter_scat = None
        self.prey_scat = None
        self.frames = []  # 保存每一帧用于生成GIF

        # 新增：用于保存从大模型得到的示例数据（用于生成数据集）
        self.dataset_samples = []
        
        # 添加日志记录器
        self.logger = None
        self.log_file = None

    def load_coordinates_from_file(self, game_index):
        """从game_coordinates.txt文件读取坐标"""
        try:
            with open('game_coordinates.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if game_index >= len(lines):
                    # 如果索引超出范围，循环使用
                    game_index = game_index % len(lines)
                line = lines[game_index].strip()
                # 解析格式: (x1, y1) (x2, y2) (x3, y3) (x4, y4)
                coords = []
                for coord_str in line.split(') ('):
                    coord_str = coord_str.strip('()')
                    x, y = map(int, coord_str.split(', '))
                    coords.append((x, y))
                return coords
        except Exception as e:
            print(f"读取坐标文件失败: {e}，使用随机坐标")
            # 如果读取失败，使用随机坐标作为备选
            return random.sample([(i, j) for i in range(20) for j in range(20)], 4)

    def manhattan_distance(self, pos1, pos2):
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_inside_triangle(self, point, triangle):
        """判断点是否在三角形内（使用重心坐标法）"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(point, triangle[0], triangle[1])
        d2 = sign(point, triangle[1], triangle[2])
        d3 = sign(point, triangle[2], triangle[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def check_win(self):
        """检查是否围捕成功"""
        # 检查所有猎人与猎物的曼哈顿距离是否都小于2
        distances = [self.manhattan_distance(hunter, self.prey) for hunter in self.hunters]

        if all(d < 2 for d in distances):
            # 检查猎物是否在猎人围成的三角形内
            if self.is_inside_triangle(self.prey, self.hunters):
                return True
        return False

    def move_agent(self, position, action, occupied_positions=None):
        """移动智能体（检查碰撞）"""
        delta = self.action_map[action]
        new_pos = (position[0] + delta[0], position[1] + delta[1])

        # 边界检查
        new_pos = (
            max(0, min(self.grid_size - 1, new_pos[0])),
            max(0, min(self.grid_size - 1, new_pos[1]))
        )

        # 碰撞检查：如果新位置被占用，则保持原位置
        if occupied_positions is not None and new_pos in occupied_positions:
            return position

        return new_pos

    def get_state_description(self):
        """获取当前状态描述"""
        state = {
            "step": self.current_step,
            "hunters": [{"id": i, "position": list(self.hunters[i])} for i in range(3)],
            "prey": {"position": list(self.prey)},
            "distances": [self.manhattan_distance(self.hunters[i], self.prey) for i in range(3)]
        }
        return state

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
- 然后以 JSON 格式返回动作结果，注意输出的json格式必须要用符号```包裹起来。
- 严禁输出 JSON 数组（例如 `[...]`），严禁输出 `action/reason` 这类字段；最终 JSON 必须是对象，且只能包含 `hunter0/hunter1/hunter2` 三个键，值必须是 0~4 的整数。
- 输出格式如下（必须严格一致）：
```json
{{"hunter0": 0, "hunter1": 0, "hunter2": 0}}
```
"""
        return prompt

    def get_llm_actions(self):
        """使用大模型获取猎人的动作"""
        prompt = self.get_prompt()

        max_retries = 5  # 最大重试次数
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.2-Exp",
                    messages=[
                        {"role": "system", "content": "你是一个智能围捕策略专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

                response_text = response.choices[0].message.content.strip()
                # print(f"LLM 响应内容:{response_text}")  # 注释掉打印
                if self.logger:
                    self.logger.debug(f"LLM 响应内容: {response_text}")

                original_response = response_text

                actions_inner = None
                if "```json" in original_response:
                    actions_inner = original_response.split("```json")[1].split("```")[0].strip()
                elif "```" in original_response:
                    actions_inner = original_response.split("```")[1].split("```")[0].strip()
                else:
                    # 尝试直接解析整个响应
                    actions_inner = original_response

                if "```" in original_response:
                    thought = original_response.split("```")[0].strip()
                else:
                    thought = original_response

                # 尝试解析JSON
                actions_dict = None
                if actions_inner is not None:
                    try:
                        actions_dict = json.loads(actions_inner)
                    except Exception:
                        try:
                            actions_dict = json.loads(actions_inner.replace("'", '"'))
                        except Exception:
                            pass

                # 如果成功解析JSON，提取动作
                if actions_dict is not None:
                    try:
                        actions = [
                            actions_dict.get("hunter0", 0),
                            actions_dict.get("hunter1", 0),
                            actions_dict.get("hunter2", 0)
                        ]
                        actions = [int(max(0, min(4, int(a)))) for a in actions]
                        
                        # 获取当前状态并保存为目标格式
                        state = self.get_state_description()
                        sample = {
                            'state': state,
                            'cot': thought if thought else "系统提示",
                            'action': json.dumps({"hunter0": actions[0], "hunter1": actions[1], "hunter2": actions[2]}, ensure_ascii=False)
                        }
                        self.dataset_samples.append(sample)
                        
                        return actions
                    except Exception as e:
                        # print(f"提取动作失败: {e}")  # 注释掉打印
                        if self.logger:
                            self.logger.warning(f"提取动作失败: {e}")
                        retry_count += 1
                        # print(f"无法解析动作，正在重试 ({retry_count}/{max_retries})...")  # 注释掉打印
                        if self.logger:
                            self.logger.warning(f"无法解析动作，正在重试 ({retry_count}/{max_retries})...")
                        continue
                else:
                    retry_count += 1
                    # print(f"无法解析JSON，正在重试 ({retry_count}/{max_retries})...")  # 注释掉打印
                    if self.logger:
                        self.logger.warning(f"无法解析JSON，正在重试 ({retry_count}/{max_retries})...")
                    continue

            except Exception as e:
                # print(f"LLM调用失败: {e}")  # 注释掉打印
                if self.logger:
                    self.logger.error(f"LLM调用失败: {e}")
                retry_count += 1
                # print(f"LLM调用异常，正在重试 ({retry_count}/{max_retries})...")  # 注释掉打印
                if self.logger:
                    self.logger.warning(f"LLM调用异常，正在重试 ({retry_count}/{max_retries})...")
                if retry_count >= max_retries:
                    raise Exception(f"达到最大重试次数 {max_retries}，仍无法获取有效决策")
                continue
        
        # 如果所有重试都失败，抛出异常
        raise Exception(f"达到最大重试次数 {max_retries}，仍无法获取有效决策")



    def step(self):
        """执行一步游戏"""
        self.current_step += 1

        hunter_actions = self.get_llm_actions()
        occupied_positions = set()
        new_hunters = []
        for i in range(3):
            new_pos = self.move_agent(self.hunters[i], hunter_actions[i], occupied_positions)
            new_hunters.append(new_pos)
            occupied_positions.add(new_pos)

        self.hunters = new_hunters

        for i in range(3):
            self.history[f'hunter_{i}'].append({
                'step': self.current_step,
                'position': self.hunters[i],
                'action': hunter_actions[i]
            })

        prey_action = random.randint(0, 4)
        self.prey = self.move_agent(self.prey, prey_action, occupied_positions)

        self.history['prey'].append({
            'step': self.current_step,
            'position': self.prey,
            'action': prey_action
        })

        return hunter_actions, prey_action

    def render(self, mode='human', save_frame=True):
        """渲染当前状态"""
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
            self.ax.set_title("Pursuit Game")

            hunter_positions = np.array([self.hunters[i] for i in range(3)])
            prey_position = np.array([self.prey])

            self.hunter_scat = self.ax.scatter(
                hunter_positions[:, 1] + 0.5,
                hunter_positions[:, 0] + 0.5,
                c='blue', s=200, label='Hunters', marker='o', edgecolors='black', linewidths=2
            )
            self.prey_scat = self.ax.scatter(
                prey_position[:, 1] + 0.5,
                prey_position[:, 0] + 0.5,
                c='red', s=300, marker='*', label='Prey', edgecolors='black', linewidths=2
            )
            self.ax.legend(loc='upper right', fontsize=12)
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.tick_params(left=False, bottom=False)

        else:
            hunter_positions = np.array([self.hunters[i] for i in range(3)])
            prey_position = np.array([self.prey])
            self.hunter_scat.set_offsets(np.c_[hunter_positions[:, 1] + 0.5,
                                               hunter_positions[:, 0] + 0.5])
            self.prey_scat.set_offsets(np.c_[prey_position[:, 1] + 0.5,
                                             prey_position[:, 0] + 0.5])

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
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            from PIL import Image
            img = Image.open(buf)
            frame = np.array(img)[:, :, :3]
            buf.close()
            self.frames.append(frame.copy())
            if self.current_step % 10 == 0:
                # print(f"已捕获第 {len(self.frames)} 帧 (步骤 {self.current_step})")  # 注释掉
                if self.logger:
                    self.logger.debug(f"已捕获第 {len(self.frames)} 帧 (步骤 {self.current_step})")

    def print_path(self):
        """打印所有智能体的决策路径"""
        print("\n" + "=" * 80)
        print("决策路径输出")
        print("=" * 80)
        for agent_name in ['hunter_0', 'hunter_1', 'hunter_2', 'prey']:
            path = self.history[agent_name]
            total_steps = len(path) - 1
            print(f"\n{agent_name} 的决策路径 (共{total_steps}步):")
            print("-" * 60)
            for record in path:
                step = record['step']
                pos = record['position']
                action = record['action']
                if action is not None:
                    action_name = self.action_names[action]
                    print(f"  步骤 {step:3d}: 位置({pos[0]:2d},{pos[1]:2d}) -> 动作: {action_name:4s}")
                else:
                    print(f"  步骤 {step:3d}: 位置({pos[0]:2d},{pos[1]:2d}) -> 动作: 初始  ")
        print("\n" + "=" * 80)

    def save_gif(self):
        """保存GIF动画"""
        if not self.frames:
            # print("❌ 没有帧可以保存")  # 注释掉
            if self.logger:
                self.logger.warning("没有帧可以保存")
            return
        if len(self.frames) < 2:
            # print(f"⚠️  只有 {len(self.frames)} 帧，无法生成动画")  # 注释掉
            if self.logger:
                self.logger.warning(f"只有 {len(self.frames)} 帧，无法生成动画")
            return
        gif_folder = "gif"
        if not os.path.exists(gif_folder):
            os.makedirs(gif_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(gif_folder, f"{timestamp}.gif")
        # print(f"\n正在保存GIF动画到: {filename}")  # 注释掉
        # print(f"   正在处理 {len(self.frames)} 帧...")  # 注释掉
        if self.logger:
            self.logger.info(f"正在保存GIF动画到: {filename}, 帧数: {len(self.frames)}")
        try:
            from PIL import Image
            pil_images = []
            for i, frame in enumerate(self.frames):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                img = Image.fromarray(frame, mode='RGB')
                pil_images.append(img)
                # if (i + 1) % 20 == 0:
                #     print(f"   处理进度: {i+1}/{len(self.frames)}")  # 注释掉
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                duration=100,
                loop=0,
                optimize=False
            )
            file_size = os.path.getsize(filename)
            # print(f"\n✅ GIF保存成功!")  # 注释掉
            # print(f"   文件路径: {filename}")  # 注释掉
            # print(f"   总帧数: {len(self.frames)}")  # 注释掉
            # print(f"   文件大小: {file_size / 1024:.2f} KB")  # 注释掉
            # print(f"   动画时长: {len(self.frames) * 0.1:.1f} 秒")  # 注释掉
            # print(f"   帧率: 10 FPS")  # 注释掉
            if self.logger:
                self.logger.info(f"GIF保存成功: {filename}, 总帧数: {len(self.frames)}, 大小: {file_size / 1024:.2f} KB")
        except Exception as e:
            # print(f"❌ 保存GIF失败: {e}")  # 注释掉
            if self.logger:
                self.logger.error(f"保存GIF失败: {e}")
            import traceback
            traceback.print_exc()

    def run(self, visualize=True):
        """运行游戏"""
        # print("=" * 50)  # 注释掉
        # print("开始围捕游戏！")  # 注释掉
        # print(f"初始位置 - 猎人: {self.hunters}, 猎物: {self.prey}")  # 注释掉
        # print("=" * 50)  # 注释掉
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info("开始围捕游戏！")
            self.logger.info(f"初始位置 - 猎人: {self.hunters}, 猎物: {self.prey}")
            self.logger.info("=" * 50)
        if visualize:
            self.render()
        while self.current_step < self.max_steps:
            hunter_actions, prey_action = self.step()
            if visualize:
                self.render()
            if self.check_win():
                # print("\n" + "=" * 50)  # 注释掉
                # print(f"🎉 围捕成功！用时 {self.current_step} 步")  # 注释掉
                # print("=" * 50)  # 注释掉
                if self.logger:
                    self.logger.info("=" * 50)
                    self.logger.info(f"围捕成功！用时 {self.current_step} 步")
                    self.logger.info("=" * 50)
                # self.print_path()  # 注释掉
                if visualize:
                    self.save_gif()
                    plt.close(self.fig)
                return True
        # print("\n" + "=" * 50)  # 注释掉
        # print(f"❌ 围捕失败！超过 {self.max_steps} 步")  # 注释掉
        # print("=" * 50)  # 注释掉
        if self.logger:
            self.logger.info("=" * 50)
            self.logger.info(f"围捕失败！超过 {self.max_steps} 步")
            self.logger.info("=" * 50)
        # self.print_path()  # 注释掉
        if visualize:
            self.save_gif()
            plt.close(self.fig)
        return False

# 运行游戏
if __name__ == "__main__":
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("统计目录", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"game_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
        ]
    )
    logger = logging.getLogger(__name__)
    
    all_samples = []
    game = None  # 保持对当前游戏对象的引用
    
    # 准备输出数据集文件（实时写入模式）
    stats_dir = os.path.join("统计目录", "json")
    os.makedirs(stats_dir, exist_ok=True)
    output_file = os.path.join(stats_dir, f"hunting_data_{timestamp}.jsonl")
    dataset_file = open(output_file, 'w', encoding='utf-8')

    try:
        # 修改点：在运行前输入游戏轮次
        print("\n" + "="*30)
        try:
            num_games_input = input("请输入要进行的游戏轮次: ").strip()
            num_games = int(num_games_input) if num_games_input else 1
        except ValueError:
            print("输入无效，默认为 1 轮")
            num_games = 1
        
        # 询问是否可视化（如果是批量生成数据，通常建议关闭）
        vis_input = input("是否开启可视化 (y/n, 建议大数据量选n): ").strip().lower()
        enable_vis = (vis_input == 'y')
        print("="*30 + "\n")
        
        logger.info(f"开始运行 {num_games} 轮游戏，可视化: {enable_vis}")
        logger.info(f"日志文件: {log_file}")
        logger.info(f"数据集输出: {output_file}")

        success_count = 0
        total_steps = 0
        
        # 使用tqdm添加进度条
        with tqdm(total=num_games, desc="游戏进度", unit="轮") as pbar:
            for i in range(num_games):
                # print(f"🚀 正在启动第 {i+1}/{num_games} 轮游戏...")  # 注释掉
                logger.info(f"启动第 {i+1}/{num_games} 轮游戏")
                pbar.set_description(f"游戏进度 [{i+1}/{num_games}]")
                
                game = HuntingGame(game_index=i)
                game.logger = logger  # 传递logger给game对象
                game.log_file = log_file
                
                result = game.run(visualize=enable_vis)
                
                # 实时写入数据到文件
                if game.dataset_samples:
                    for sample in game.dataset_samples:
                        dataset_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    dataset_file.flush()  # 强制刷新缓冲区，确保实时写入
                    
                    all_samples.extend(game.dataset_samples)
                    game.dataset_samples = [] # 清空以防内存积压

                if result:
                    success_count += 1
                    total_steps += game.current_step
                
                # print(f"✅ 第 {i+1} 轮结束。目前共收集样本: {len(all_samples)} 条\n")  # 注释掉
                logger.info(f"第 {i+1} 轮结束，共收集样本: {len(all_samples)} 条")
                pbar.update(1)
                pbar.set_postfix({"样本数": len(all_samples), "成功": success_count})

        print("\n" + "=" * 50)
        print(f"全部测试结束！")
        print(f"总轮次: {num_games}")
        print(f"成功率: {success_count}/{num_games} ({success_count/num_games*100:.1f}%)")
        if success_count > 0:
            print(f"平均成功步数: {total_steps/success_count:.1f}")
        print(f"总样本数: {len(all_samples)}")
        print("=" * 50)
        
        logger.info(f"全部测试结束: {num_games}轮, 成功率: {success_count}/{num_games}")
        logger.info(f"总样本数: {len(all_samples)}")

    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断 (Ctrl+C)！正在保存已收集的数据...")
        logger.warning("程序被用户中断")
        # 尝试抢救当前正在运行的游戏中的数据
        if game is not None and game.dataset_samples:
            # print(f"   正在合并最后一次游戏的 {len(game.dataset_samples)} 条数据...")  # 注释掉
            logger.info(f"合并最后一次游戏的 {len(game.dataset_samples)} 条数据")
            for sample in game.dataset_samples:
                dataset_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
            dataset_file.flush()
            all_samples.extend(game.dataset_samples)

    finally:
        # 关闭数据集文件
        dataset_file.close()
        
        # 在统计目录下生成 json 数据集
        if all_samples:
            print(f"\n✅ 数据集已实时保存到: {output_file}")
            print(f"   总样本数: {len(all_samples)}")
            print(f"   日志文件: {log_file}")
            logger.info(f"数据集保存成功: {output_file}, 共 {len(all_samples)} 条样本")
        else:
            print("\n⚠️  未采集到任何样本。")
            logger.warning("未采集到任何样本")

        print('\n程序结束')
        logger.info("程序结束")