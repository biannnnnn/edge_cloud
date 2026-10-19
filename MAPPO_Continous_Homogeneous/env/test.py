import numpy as np
from pettingzoo.utils import random_demo
from pursuit_env import PursuitEnv  # 假设前面的环境代码保存在 pursuit_env.py 中

def test_pursuit_env(episodes=1, render=True):
    """
    测试 PursuitEnv 环境，运行指定数量的回合。
    参数：
        episodes: 运行的回合数
        render: 是否渲染环境
    """
    env = PursuitEnv(grid_size=20, max_steps=100)
    
    for episode in range(episodes):
        print(f"\n=== 回合 {episode + 1} ===")
        observations, infos = env.reset(seed=episode)
        
        # 打印初始观测
        print("初始观测:")
        for agent, obs in observations.items():
            print(f"{agent}: {obs}")
        
        done = False
        step = 0
        while not done:
            # 为每个智能体随机选择动作
            actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
            print(f"\n步数 {step + 1} 动作: {actions}")
            
            # 执行一步
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 渲染环境
            if render:
                env.render()
            
            # 打印奖励和观测
            print("奖励:", rewards)
            print("观测:")
            for agent, obs in observations.items():
                print(f"{agent}: {obs}")
            
            # 检查是否终止
            done = all(terminations.values())
            step += 1
            
            # 如果环境终止，打印最终结果
            if done:
                print("\n游戏结束！")
                if all(abs(env.positions[f'hunter_{i}'][0] - env.positions['prey'][0]) + 
                       abs(env.positions[f'hunter_{i}'][1] - env.positions['prey'][1]) < 2 
                       for i in range(3)):
                    print("围捕成功！猎人获胜！")
                else:
                    print("围捕失败！猎物逃脱！")
        
        # 打印最终奖励
        print("最终奖励:", rewards)
    
    env.close()

if __name__ == "__main__":
    # 运行一个回合，开启渲染
    test_pursuit_env(episodes=1, render=True)