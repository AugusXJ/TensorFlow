"""
TensorFlow实战
4.2 实现自编码器
"""
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 参数初始化方法xavier initialization
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveFaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """
        去燥编码器
        :param n_input: 输入变量数
        :param n_hidden: 隐藏节点数
        :param transfer_function: 隐含层激活函数
        :param optimizer: 优化函数，默认为Adam
        :param scale: 高斯噪声系数
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weight = self._initializer_weights()
        self.weights = network_weight
        # 网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
            self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 损失函数
        # tf.substract 求差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 权重初始化
    def _initializer_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))     # 自编码器输入层节点数量和输出层相同
        return all_weights

    # 训练数据获得损失cost
    def partial_fit(self, x):
        loss_cost, opt = self.sess.run((self.cost, self.optimizer),
                                       feed_dict={self.x: x, self.scale: self.training_scale})
        return loss_cost

    # 检验数据获得cost
    def calc_total_cost(self, x):
        return self.sess.run(self.cost, feed_dict={self.x: x, self.scale: self.training_scale})

    # 输出隐藏层后的特征
    def transform(self, x):
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.scale: self.training_scale})

    # 隐藏层输出
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 运行隐层和输出层
    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x, self.scale: self.training_scale})

    # 获取隐层权重w1
    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐层的篇制系数b1
    def get_biases(self):
        return self.sess.run(self.weights['b1'])


# 使用sklearn对数据进行标准化处理
def standard_scale(x_train, x_test):
    preprocessor = prep.StandardScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test


# 从训练集中选取batch_size的样本
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)       # 读取数据
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)       # 训练样本数
    training_epochs = 100        # 训练轮数
    batch_size = 128
    display_step = 1
    autoencoder = AdditiveFaussianNoiseAutoencoder(n_input=784,                                     # 创建自编码器
                                                   n_hidden=200,
                                                   transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.01)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
