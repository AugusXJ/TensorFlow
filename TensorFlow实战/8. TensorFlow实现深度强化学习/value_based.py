"""
Tensorflow实战，强化学习，value-based
"""

import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import os
import time

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        """
        物体对象类
        :param coordinates: x，y坐标值
        :param size: 尺寸
        :param intensity: 亮度值 
        :param channel: RGB颜色通道
        :param reward: 奖励值
        :param name: 名称
        """
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class gameEnv():
    def __init__(self, size):
        """
        环境类
        :param size: 环境尺寸   [integer. integer]
        """
        self.sizex = size
        self.sizey = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        plt.imshow(a, interpolation="nearest")
        pass

    def reset(self):
        """
        环境重置
        :return: 当前状态
        """
        self.objects = []
        hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')        # 英雄对象
        self.objects.append(hero)
        goal = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')           # 目标对象
        self.objects.append(goal)
        hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')          # 火 对象
        self.objects.append(hole)
        goal2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')          # 目标对象2
        self.objects.append(goal2)
        hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')          # 火 对象2
        self.objects.append(hole2)
        goal3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')          # 目标对象3
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')          # 目标对象4
        self.objects.append(goal4)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self, direction):
        """
        移动角色
        :param direction: 方向
        :return: None
        """
        hero = self.objects[0]
        herox = hero.x
        heroy = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizey-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizex-2:
            hero.x += 1
        self.objects[0] = hero

    def newPosition(self):
        """
        从现有空位选择一个位置
        :return: direction  [integer, integer]
        """
        iterables = [range(self.sizex), range(self.sizey)]
        points = []
        for t in itertools.product(*iterables):     # itertools.product可以获得几个变量的所有组合
            points.append(t)
        current_positions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in current_positions:
                current_positions.append((objectA.x, objectA.y))
        for pos in current_positions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        """
        判断hero是否触碰goal or fire
        :return: Reward, if crashed (float, boolean)
        """
        others = []
        hero = None
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
                return other.reward, True
        return 0.0, False

    def renderEnv(self):
        """
        刷新环境
        :return: ndarray 
        """
        a = np.ones([self.sizey + 2, self.sizex + 2, 3])        # 图片size
        a[1:-1, 1:-1, :] = 0                                    # 边框
        for item in self.objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)
        return a

    def step(self, action):
        self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        return state, reward, done

class Qnetwork():
    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = tf.contrib.layers.convolution2d(
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
            padding='VALID', biases_initializer=None
        )
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None
        )
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None
        )
        self.conv4 = tf.contrib.layers.convolution2d(
            inputs=self.conv3, num_outputs=512, kernel_size=[7, 7], stride=[1, 1],
            padding='VALID', biases_initializer=None
        )
        # Duel DQN
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)       # 拆分成2段，维度是第三维
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size//2, env.actions]))       # streamA全连接层权重
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamA, self.VW)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1,
                                                                            keep_dims=True))    # 合并Q值
        self.predict = tf.argmax(self.Qout, 1)
        # Double DQN
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        # loss
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size=50000):
        """
        经验回放
        :param buffer_size: 
        """
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def processState(states):
    """
    将84x84x3的states扁平化为1维向量
    :param states: 
    :return: 
    """
    return np.reshape(states, [21168])

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value() * tau) +
                                                            (1-tau) * tfVars[idx + total_vars//2].value()))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

batch_size = 32
update_freq = 4             # 每隔多少步更新一次模型参数
y = .99                     # Q值得衰减系数
startE = 1.                  # 初始时执行随机行为的概率
endE = 0.1                  # 最终执行随机行为的概率
anneling_steps = 10000.     # 从初始随机行为到最终随机行为所需步数
num_episodes = 10000        # 总共需要多少次游戏
pre_train_steps = 10000     # 正式使用DQN选择action前需要多少步随机action
max_epLength = 50           # 每个episode进行多少步action
load_model = False          # 是否读取之前训练的模型
path = "./dqn"              # 模型存储的路径
h_size = 512                # DQN网络最后的全连接层隐含节点数
tau = 0.001                 # target DQN向主DQN学习的速率


if __name__ == '__main__':
    env = gameEnv(size=5)
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)
    myBuffer = experience_buffer()                          # 创建buffer对象
    e = startE
    stepDrop = (startE - endE) / anneling_steps

    rList = []
    total_steps = 0

    saver = tf.train.Saver()
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if load_model is True:
            print('Loading Model ......')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(init)
        updateTarget(targetOps, sess)
        for i in range(num_episodes+1):
            episodeBuffer = experience_buffer()
            s = env.reset()
            s = processState(s)
            d = False                       # done标记
            rAll = 0                        # 总reward
            j = 0                           # 步数
            while j < max_epLength:
                j += 1
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0, 4)
                else:
                    a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                s1, r, d = env.step(a)
                s1 = processState(s1)
                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    if total_steps % update_freq == 0:                                # 开始训练
                        trainBatch = myBuffer.sample(batch_size)
                        A = sess.run(mainQN.predict,
                                     feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})       # 主模型的action
                        Q = sess.run(targetQN.Qout,
                                     feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})     # 所有action的Q
                        doubleQ = Q[range(batch_size), A]
                        targetQ = trainBatch[:, 2] + y * doubleQ
                        _ = sess.run(mainQN.updateModel, feed_dict={
                            mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                            mainQN.targetQ: targetQ,
                            mainQN.actions: trainBatch[:, 1]
                        })
                        updateTarget(targetOps, sess)
                rAll += r
                s = s1
                if d is True:
                    break
            myBuffer.add(episodeBuffer.buffer)
            rList.append(rAll)
            if i > 0 and i % 25 == 0:
                print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25:]))
            if i > 0 and i % 1000 == 0:
                saver.save(sess, path + '/model-' + str(i) + '.cpkt')
                print("Saved Model")
        saver.save(sess, path + 'model-' + str(i) + '.cpkt')