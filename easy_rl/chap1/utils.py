import gym # 导入 Gym 的 Python 接口环境包
import time

def random_agent():
    env = gym.make('CartPole-v0') # 构建实验环境
    env.reset() # 重置一个回合
    for i in range(100):
        env.render() # 显示图形界面
        action = env.action_space.sample() # 从动作空间中随机选取一个动作
        observation = env.observation_space.sample()
        # print(observation)
        if action == 0:
            print(f"step {i}, 向左")
        else:
            print(f"step {i}, 向右")
        observation, reward, done, info =env.step(action) # 用于提交动作，括号内是具体的动作
        print(f'observation: {observation}, reward: {reward}, done: {done}, info: {info}')
        time.sleep(0.1)
    env.close() # 关闭环境
def print_all_envs():
    from gym import envs
    env_specs = envs.registry.all()
    envs_ids = [env_spec.id for env_spec in env_specs]
    print(envs_ids)
def print_mountain_car_env():
    import gym
    env = gym.make('MountainCar-v0')
    print('观测空间 = {}'. format(env.observation_space))
    print('动作空间 = {}'. format(env.action_space))
    print('观测范围 = {} ~ {}'. format(env.observation_space.low,
    env.observation_space.high))
    print('动作数 = {}'. format(env.action_space.n))

class SimpleAgent:
    def __init__(self, env):
        pass
    def decide(self, observation): # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
        0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作
    def learn(self, *args): # 学习
        pass


def play(env, agent, render=False, train=False):
    episode_reward = 0. # 记录回合总奖励，初始值为0
    observation = env.reset() # 重置游戏环境，开始新回合
    while True: # 不断循环，直到回合结束
        if render: # 判断是否显示
            env.render() # 显示图形界面
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action) # 执行动作
        episode_reward += reward # 收集回合奖励
        if train: # 判断是否训练智能体
            agent.learn(observation, action, reward, done) # 学习
        if done: # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward # 返回回合总奖励

def do_play_mountain_car():
    print_mountain_car_env()
    env = gym.make('MountainCar-v0')
    agent = SimpleAgent(env)
    env.seed(3) # 设置随机种子，让结果可复现
    episode_reward = play(env, agent, render=True)
    print('回合奖励 = {}'. format(episode_reward))
    env.close() # 关闭图形界面
    import numpy as np
    episode_rewards = [play(env, agent) for _ in range(100)]
    print('平均回合奖励 = {}'. format(np.mean(episode_rewards)))
