"""
13年版本DQN玩CartPole
"""

# import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from retro_contest.local import make

ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10  # experience replay buffer size
BATCH_SIZE = 8  # size of minibatch


class DQN:
    # DQN agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape
        self.state_dim_row = env.observation_space.shape[0]
        self.state_dim_col = env.observation_space.shape[1]
        self.state_dim_channel = env.observation_space.shape[2]
        self.action_dim = env.action_space.n
        self.state_input = tf.placeholder(tf.float32,
                                          [None,
                                           self.state_dim_row,
                                           self.state_dim_col,
                                           self.state_dim_channel])      # 输入空间
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])  # 动作空间
        self.keep_prob = tf.placeholder("float")

        self.Q_value = self.create_q_network()                                  # 神经网络计算的Q值
        self.y_input = tf.placeholder("float", [None])                          # targetQ值
        self.optimizer = self.create_training_method()                          # 训练启动器

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    # 卷积和池化
    @staticmethod
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def create_q_network(self):
        # network weights
        # 卷积1
        w_conv1 = self.weight_variable([5, 5, 3, 33])
        b_conv1 = self.bias_variable([33])
        h_conv1 = tf.nn.relu(self.conv2d(self.state_input, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # 卷积2
        w_conv2 = self.weight_variable([5, 5, 33, 66])
        b_conv2 = self.bias_variable([66])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # hidden layers
        input_dim = int(self.state_dim_row / 4) * int(self.state_dim_col / 4) * 66
        w_fc1 = self.weight_variable([input_dim, 256])
        b_fc1 = self.bias_variable([256])
        h_pool2_flat = tf.reshape(h_pool2, [-1, input_dim])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        # DropOut
        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # 输出层
        w_fc2 = self.weight_variable([256, self.action_dim])
        b_fc2 = self.bias_variable([self.action_dim])
        y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        # Q Value layer
        return y_conv

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_training_method(self):
        q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        return tf.train.AdamOptimizer(0.0001).minimize(cost)

    def perceive(self, state, action, reward, next_state, done):
        """
        存储信息
        :param state: 
        :param action: 
        :param reward: 
        :param next_state: 
        :param done: 
        :return: 
        """
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_q_network()

    def train_q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state],
            self.keep_prob: 1.0
        })[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value)
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])


def main():
    # initialize OpenAI Gym env and dqn agent
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = np.array(env.reset())
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            print(action)
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(np.array(state))  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            if ave_reward >= 200:
                break
if __name__ == '__main__':
    main()
