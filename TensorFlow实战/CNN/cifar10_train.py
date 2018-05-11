import CNN.cifar10 as cifar10
import CNN.cifar10_input as cifar10_input
import numpy as np
import tensorflow as tf
import time

print('-------------------------------------------')
max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batched-bin'

# 权重初始化
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2.loss(var), w1, name='weight_loss')
        tf.add_to_clollection('losses', weight_loss)
    return var

cifar10.maybe_download_and_extract()

# 获取数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 输入数据的placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])      # 图片规格24x24x3
label_holder = tf.placeholder(tf.float32, [batch_size])

# 卷积层1
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=0.05, w1=0.0)   # 设置初始化函数标准差为0.05，不进行正则
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')     # 卷积操作
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias=bias1))         # ReLU激活函数
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')     # 池化操作
norm1 = tf.nn.lrn(pool1, 4, bias=bias1, alpha=0.001/9.0, beta=0.75)      # LRN层

# 卷积层2
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=0.05, w1=0.1)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias=bias2))                         # 图形用bias_add
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第三层：全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)        # 隐节点个数384
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3)+bias3)

# 第四层：全连接层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.1)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5) + bias5)

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('lossed', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss(logits, label_holder)                           # 损失函数
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)      # 优化器

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer()
tf.train.start_queue_runners()

# 开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = 'step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)'
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))


