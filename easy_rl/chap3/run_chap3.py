from chap3.utils import *
def run_chap3():
    # 获取参数
    cfg = Config() 
    # 训练
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    # 可视化Q表
    agent.visualize_q_table()
    plot_rewards(res_dic['rewards'], title=f"training curve of {cfg.algo_name} for {cfg.env_name}")  
    # 测试
    res_dic = test(cfg, env, agent)
    plot_rewards(res_dic['rewards'], title=f"testing curve of {cfg.algo_name} for {cfg.env_name}")  # 画出结果
