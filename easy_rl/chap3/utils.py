import gym

import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from chap3.qlearning import QLearning
from chap3.cliffwalkingwrapper import CliffWalkingWapper

def train(cfg,env,agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}')
    # print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset(seed=cfg.seed)  # 重置环境,即开始新的回合
        while True:
            action = agent.sample_action(state)  # 根据算法采样一个动作
            next_state, reward, terminated, info = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, terminated)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        #agent.visualize_q_table()
        time.sleep(1)
        if (i_ep+1)%20==0:
            print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")
    print('完成训练！')
    return {"rewards":rewards}
def save_trajectories(all_states):
    # 将状态序列保存到CSV文件
    import pandas as pd
    import datetime
    
    # 获取当前时间作为文件名的一部分
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'./data/test_trajectories_{current_time}.csv'
    
    # 找到最长的状态序列长度
    max_length = max(len(states) for states in all_states)
    
    # 将所有状态序列填充到相同长度
    padded_states = [states + [-1] * (max_length - len(states)) for states in all_states]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(padded_states)
    df.to_csv(filename, index=False, header=[f'Step_{i}' for i in range(max_length)])
    print(f'轨迹已保存至：{filename}')
def test(cfg,env,agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}')
    rewards = []  # 记录所有回合的奖励
    all_states = []  # 记录所有回合的状态序列
    
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset(seed=cfg.seed)  # 重置环境
        states_list = [state]  # 记录当前回合的状态序列
        
        while True:
            action = agent.predict_action(state)  # 根据算法选择一个动作
            next_state, reward, terminated, info = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            states_list.append(state)  # 记录状态
            ep_reward += reward
            if terminated:
                break
        
        rewards.append(ep_reward)
        all_states.append(states_list)  # 将当前回合的状态序列添加到总列表中
        print(f"回合数：{i_ep+1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    
    save_trajectories(all_states)
    print('完成测试！')
    return {"rewards":rewards}
def env_agent_config(cfg,seed=1):
    '''创建环境和智能体
    '''    
    env = gym.make(cfg.env_name,new_step_api=True)  
    env = CliffWalkingWapper(env)
    n_states = env.observation_space.n # 状态维度
    n_actions = env.action_space.n # 动作维度
    print(f'observation {env.observation_space}, \n action {env.action_space}')
    agent = QLearning(n_states,n_actions,cfg)
    return env,agent
class Config:
    '''配置参数
    '''
    def __init__(self):
        self.env_name = 'CliffWalking-v0' # 环境名称
        self.algo_name = 'Q-Learning' # 算法名称
        self.train_eps = 400#400 # 训练回合数
        self.test_eps = 400 # 测试回合数
        self.max_steps = 200 # 每个回合最大步数
        self.epsilon_start = 0.95 #  e-greedy策略中epsilon的初始值
        self.epsilon_end = 0.01 #  e-greedy策略中epsilon的最终值
        self.epsilon_decay = 300 #  e-greedy策略中epsilon的衰减率
        self.gamma = 0.9 # 折扣因子
        self.lr = 0.1 # 学习率
        self.seed = 1 # 随机种子
def smooth(data, weight=0.9):  
    '''用于平滑曲线
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed
def plot_rewards(rewards,title="learning curve"):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlim(0, len(rewards))  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()
