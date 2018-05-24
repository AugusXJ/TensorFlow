"""
Tensorflow实战，强化学习，policy-based
"""
import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')
H = 50                     # 隐层节点数
batch_size = 25
learning_rate = 1e-1       # 学习率
D = 4                       # 输入层节点数量
gama = 0.99

# 双层神经网络，输入状态，输出行为概率
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)      # 优化算法adam


def discount_rewards(r):
    """
    估算每个action对应的潜在价值
    :param r: 连续决策的直接受益
    :return: 连续决策的潜在受益
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gama + r[t]
        discounted_r[t] = running_add
    return discounted_r

# 计算损失函数
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
loglike = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglike * advantages)            # 损失函数

# 计算梯度
tvars = tf.trainable_variables()            # 获取全部可训练参数，一个list的形式返回W1，W2
newGrads = tf.gradients(loss, tvars)        # 计算梯度

# 使用adam更新梯度
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

xs, ys, drs = [], [], []        # xs：环境信息列表 ys：定义的label列表 drs：记录每个行为的收益
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0               # 收集参数梯度
    while episode_number <= total_episodes:
        if reward_sum/batch_size > 150 or rendering is True:
            env.render()
            rendering = False
        x = np.reshape(observation, [1, D])
        tfprob = sess.run(probability, feed_dict={observations: x})         # 获取行为概率
        action = 1 if np.random.uniform() < tfprob else 0                   # 根据概率确定行为
        xs.append(x)                                                        # append状态
        y = 1 - action
        ys.append(y)
        observation, reward, done, info = env.step(action)                  # 执行动作
        reward_sum += reward                                                # 增加累积收益
        drs.append(reward)                                                  # append收益
        if done:                                                            # 如果一次实验结束
            episode_number += 1                                             # 增次episode
            epx = np.vstack(xs)                                             # 将连续状态压成一列
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []                                        # 重置状态、标签和收益
            discounted_epr = discount_rewards(epr)                          # 计算潜在价值
            discounted_epr -= np.mean(discounted_epr)                       # 标准化
            discounted_epr /= np.std(discounted_epr)                        # 标准化
            # 输入神经网络
            tGard = sess.run(newGrads, feed_dict={
                observations: epx, input_y: epy, advantages: discounted_epr})  # 使用3个ep来计算梯度
            for ix, grad in enumerate(tGard):
                gradBuffer[ix] += grad
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                print('Average reward for episode %d : %f.' % (episode_number, reward_sum/batch_size))
                if reward_sum/batch_size > 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break
                reward_sum = 0
            observation = env.reset()
