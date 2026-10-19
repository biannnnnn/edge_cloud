from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy


class PursuitEnv(ParallelEnv):
    metadata = {
        'name': 'pursuit_v0',
        'render_modes': ['human'],
    }

    def __init__(self, grid_size=20, max_steps=100, render_mode=None):
        """
        自定义PettingZoo围捕游戏环境：
        - 3个猎人（强化学习控制）追捕1个猎物（GLM-4控制）。
        - 图形渲染：matplotlib动态显示猎人(H)和猎物(P)的位置。
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = ['hunter_0', 'hunter_1', 'hunter_2', 'prey']
        # self.agents = self.possible_agents
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self.action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(low=0, high=self.grid_size - 1, shape=(8,), dtype=int)
            for agent in self.possible_agents
        }

        # 用于绘图的句柄
        self.fig = None
        self.ax = None
        self.hunter_scat = None
        self.prey_scat = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0

        # 随机放置智能体，确保不重叠
        positions = np.random.choice(self.grid_size * self.grid_size, len(self.agents), replace=False)
        self.positions = {}
        for i, agent in enumerate(self.agents):
            x = positions[i] // self.grid_size
            y = positions[i] % self.grid_size
            self.positions[agent] = (x, y)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        self.steps += 1

        action_deltas = {
            0: (0, 0),   # 无动作
            1: (0, -1),  # 上
            2: (0, 1),   # 下
            3: (-1, 0),  # 左
            4: (1, 0),   # 右
        }
        if isinstance(actions, dict):
            for agent, action in actions.items():
                dx, dy = action_deltas.get(action, (0, 0))
                x, y = self.positions[agent]
                new_x = max(0, min(self.grid_size - 1, x + dx))
                new_y = max(0, min(self.grid_size - 1, y + dy))
                self.positions[agent] = (new_x, new_y)
        elif isinstance(actions, list) or isinstance(actions, np.ndarray):
            for agent, action in enumerate(actions):
                agent = self.agents[agent]
                dx, dy = action_deltas.get(action, (0, 0))
                x, y = self.positions[agent]
                new_x = max(0, min(self.grid_size - 1, x + dx))
                new_y = max(0, min(self.grid_size - 1, y + dy))
                self.positions[agent] = (new_x, new_y)

        prey_pos = self.positions['prey']
        hunter_positions = [self.positions[f'hunter_{i}'] for i in range(3)]
        distances = [abs(hx - prey_pos[0]) + abs(hy - prey_pos[1]) for hx, hy in hunter_positions]

        # 计算是否在三角形内部
        in_triangle = self._is_point_in_triangle(prey_pos, *hunter_positions)

        # 围捕成功条件
        captured = all(d < 2 for d in distances) and in_triangle
        timeout = self.steps > self.max_steps

        terminations = {agent: captured for agent in self.agents}
        truncations = {agent: timeout for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}

        # 计算团队奖励（全局共享）
        team_reward = -1  # 时间惩罚（每步）

        sum_dist = sum(distances)
        team_reward += -0.2 * sum_dist  # 距离惩罚

        # if in_triangle:
        #     area = self._calculate_triangle_area(*hunter_positions)
        #     area_reward = max(0, 10 - area)
        #     area_reward = min(area_reward, 10)
        #     team_reward += area_reward  # 包围奖励

        if captured:
            team_reward += 100  # 成功奖励

        # 如果静止不动，进行惩罚，0.1
        # 分配奖励：猎人共享团队奖励，猎物得到负值
        for i in range(3):
            rewards[f'hunter_{i}'] = team_reward
        rewards['prey'] = -team_reward

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if captured or timeout:
            self.agents = []

        if self.render_mode == 'human':
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        pos_list = []
        for a in self.possible_agents:
            pos_list.extend(self.positions[a])
        return np.array(pos_list, dtype=int)

    def _is_point_in_triangle(self, p, a, b, c):
        """
        检查点 p 是否在三角形 a, b, c 内部（包括边界）。
        使用重心坐标法。
        """
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _calculate_triangle_area(self, a, b, c):
        """
        计算三角形面积，使用鞋带公式。
        """
        return abs(
            (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0
        )

    def render(self, mode='human'):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

            # ✅ 网格范围覆盖完整格子（0 ~ grid_size）
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)

            # ✅ 网格线画在整数处，格心坐标是 (i+0.5, j+0.5)
            self.ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
            self.ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
            self.ax.grid(True, color='gray', linewidth=0.5, alpha=0.6)

            # ✅ 保持方形比例、翻转y轴使(0,0)在左上角
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()

            self.ax.set_title("Pursuit Game")

            # 初始点位：绘制时偏移 +0.5，使其居中到格子内
            hunter_positions = np.array([self.positions[f'hunter_{i}'] for i in range(3)])
            prey_position = np.array([self.positions['prey']])

            self.hunter_scat = self.ax.scatter(
                hunter_positions[:, 1] + 0.5,
                hunter_positions[:, 0] + 0.5,
                c='blue', s=180, label='Hunters'
            )
            self.prey_scat = self.ax.scatter(
                prey_position[:, 1] + 0.5,
                prey_position[:, 0] + 0.5,
                c='red', s=220, marker='*', label='Prey'
            )

            self.ax.legend(loc='upper right')

            # ✅ 可选：隐藏坐标轴刻度标签（保留格线）
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.tick_params(left=False, bottom=False)

        else:
            hunter_positions = np.array([self.positions[f'hunter_{i}'] for i in range(3)])
            prey_position = np.array([self.positions['prey']])

            # ✅ 更新时同样加0.5偏移
            self.hunter_scat.set_offsets(np.c_[hunter_positions[:, 1] + 0.5,
                                               hunter_positions[:, 0] + 0.5])
            self.prey_scat.set_offsets(np.c_[prey_position[:, 1] + 0.5,
                                             prey_position[:, 0] + 0.5])

        self.ax.set_title(f"Step: {self.steps}")
        plt.pause(0.1)
        plt.draw()

    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None

# 示例用法：
if __name__ == "__main__":
    env = PursuitEnv(grid_size=10, max_steps=50, render_mode='human')
    obs, infos = env.reset()

    for _ in range(100):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if not env.agents:
            break
    env.close()