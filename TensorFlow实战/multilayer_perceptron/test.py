from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()
#   给隐含层的参数设置Variable并进行初始化，这里in_units是输入节点数，h1_units即隐含层的输出节点数设为300。
#   由于模型使用的激活函数是ReLU，所以需要使用正态分布给参数加一点噪声，来打破完全对称并且避免0梯度。
in_units=784
h1_units=300
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))
#   接下来定义输入x的占位符，在训练和预测时，Dropout的比例keep_prob（即保留节点的概率）是不一样的，训练时小于1，预测时等于1
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)
#   接下来定义模型结构，首先定义一个命名为hidden1的实现一个激活函数为ReLu的隐含层，接下来实现Dropout功能，最后是softmax输出层。
hidden1=tf.nn.relu(tf.matmul(x,W1) + b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
#   接下来定义损失函数和选择优化器来优化loss，这里的损失函数继续使用交叉信息熵，优化器选择自适应的优化器Adagrad，学习速率设为3。
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#   训练步骤
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
#   对模型进行准确率评测
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))