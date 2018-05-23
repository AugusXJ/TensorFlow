"""
TensorFlow实战
4.4 实现多层感知机
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()      # 默认session
# 第一步 变量定义
in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal(shape=[in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros(shape=[h1_units, 10]))
b2 = tf.Variable(tf.zeros(shape=[10]))

x = tf.placeholder(tf.float32, [None, in_units])  # None 为样本数量
y_ = tf.placeholder(tf.float32, shape=[None, 10])   # 样本标签
keep_prob = tf.placeholder(tf.float32)          # dropout节点丢失的比例

# 第二步 损失函数和优化算法
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)         # 隐藏层
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))       # 损失函数交叉熵
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)                    # 优化算法

# 第三步 训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 第四步 验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # 求正确率
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
