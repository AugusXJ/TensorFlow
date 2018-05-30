"""
Tensorflow实战，强化学习，value-based
"""

import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os

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
                others.append(hero)
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
        self.conv1 = tf.contrib.layers.convolition2d(
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
            padding='VALID', biases_initializer=None
        )
        self.conv2 = tf.contrib.layers.convolition2d(
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None
        )
        self.conv3 = tf.contrib.layers.convolition2d(
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None
        )
        self.conv4 = tf.contrib.layers.convolition2d(
            inputs=self.conv3, num_outputs=512, kernel_size=[7, 7], stride=[1, 1],
            padding='VALID', biases_initializer=None
        )
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)       # 拆分成2段，维度是第三维
        self.streamS = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)




if __name__ == '__main__':
    env = gameEnv(size=5)