图像特征点提取这个任务，其实没有那么困难，因为他并没有延迟奖励的问题。而强化学习最难的是延迟奖励。

为什么我们关注强化学习，其中非常重要的一个原因就是强化学习得到的模型可以有超人类的表现

强化学习里面一个重要的课题就是近期奖励和远期奖励的权衡（trade-off），研究怎么让智能体取得更多的远期奖励。

状态是对世界的完整描述，不会隐藏世界的信息。观测是对状态的部分描述，可能会遗漏一些信息

通常情况下，强化学习一般使用随机性策略，随机性策略有很多优点。比如，在学习时可以通过引入一定的随机性来更好地探索环境；随机性策略的动作具有多样性，这一点在多个智能体博弈时非常重要。采用确定性策略的智能体总是对同样的状态采取相同的动作，这会导致它的策略很容易被对手预测。

基于价值迭代的方法只能应用在不连续的、离散的环境下（如围棋或某些游戏领域），对于动作
集合规模庞大、动作连续的场景（如机器人控制领域），其很难学习到较好的结果（此时基于策略迭代的
方法能够根据设定的策略来选择连续的动作）

测试智能体在 Gym 库中某个任务的性能时，出于习惯使然，学术界一般最关心 100 个回合的平均回
合奖励。对于有些任务，还会指定一个参考的回合奖励值，当连续 100 个回合的奖励大于指定的值时，则
认为该任务被解决了。而对于没有指定值的任务，就无所谓任务被解决了或没有被解决
