# 下载手写数字数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()      # 默认session
x = tf.placeholder(tf.float32, [None, 784])   # 存放图片特征
W = tf.Variable(tf.zeros([784, 10]))    # W变量
b = tf.Variable(tf.zeros([10]))         # b变量
y = tf.nn.softmax(tf.matmul(x, W) + b)              # 单层神经网络

# 损失函数
y_ = tf.placeholder(tf.float32, [None, 10])     # 存放图片标签
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))       # 损失函数

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)             # 梯度下降
tf.global_variables_initializer().run()  # 初始化变量
for i in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 计算错误率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))

