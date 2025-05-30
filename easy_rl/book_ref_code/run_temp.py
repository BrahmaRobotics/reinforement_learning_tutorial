import gym # 导入 Gym 的 Python 接口环境包
import time
env = gym.make('CartPole-v0') # 构建实验环境
env.reset() # 重置一个回合
for i in range(200):
    env.render() # 显示图形界面
    action = env.action_space.sample() # 从动作空间中随机选取一个动作
    print(f"step {i}, action: {action}")
    action = 0
    env.step(action) # 用于提交动作，括号内是具体的动作
    time.sleep(0.1)
env.close() # 关闭环境