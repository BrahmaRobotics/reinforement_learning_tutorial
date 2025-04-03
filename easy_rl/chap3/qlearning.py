import numpy as np
import math
from collections import defaultdict
import time
class QLearning(object):
    def __init__(self,n_states,
                 n_actions,cfg):
        self.n_actions = n_actions 
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma  
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(n_actions)) # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射，即Q表
    def sample_action(self, state):
        ''' 采样动作，训练时用
        '''
        # 计数器加1，用于追踪采样次数
        self.sample_count += 1
        # 计算当前的epsilon值，随着训练进行会逐渐减小
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) # epsilon是会递减的，这里选择指数递减
        # e-greedy 策略
        if np.random.uniform(0, 1) > self.epsilon:
            # 以1-epsilon的概率选择最优动作（利用）
            action = np.argmax(self.Q_table[str(state)]) # 选择Q(s,a)最大对应的动作
        else:
            # 以epsilon的概率随机选择动作（探索）
            action = np.random.choice(self.n_actions) # 随机选择动作
        return action
    def predict_action(self,state):
        ''' 预测或选择动作，测试时用
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action
    def update(self, state, action, reward, next_state, terminated):
        Q_predict = self.Q_table[str(state)][action] 
        if terminated: # 终止状态
            Q_target = reward  
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
    def visualize_q_table(self):
        '''可视化Q表
        将Q表转换为便于理解的网格形式，展示每个状态下各个动作的Q值
        动作分别为：上(0)、右(1)、下(2)、左(3)
        '''
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 设置图片清晰度
        #plt.rcParams['figure.dpi'] = 300

        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用黑体字体
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建一个网格形状的Q值数组 (4x12的网格，每个格子有4个动作)
        grid_q_table = np.zeros((4, 12, 4))
        
        # 将字典形式的Q表转换为网格形式
        for state in range(4 * 12):  # 总共48个状态
            row = 3 - state // 12    # 反转行索引使得原点在左下角
            col = state % 12
            grid_q_table[row, col] = self.Q_table[str(state)]
        
        # 创建子图，展示四个动作的热力图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        action_names = ['上', '右', '下', '左']
        
        for i, ax in enumerate(axes.flat):
            sns.heatmap(grid_q_table[:,:,i], ax=ax, cmap='RdYlBu_r', 
                       annot=True, fmt='.2f', cbar=True)
            ax.set_title(f'动作 {action_names[i]} 的Q值')
            ax.set_xlabel('列')
            ax.set_ylabel('行')
        
        plt.tight_layout()
        plt.savefig(f'./data/q_table_{self.sample_count}.png')
        #plt.show()