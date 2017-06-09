# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)

X = tf.placeholder("float", shape=[None,784])
Y = tf.placeholder("float",shape=[None,10])
X_img = tf.reshape(X,[-1,28,28,1])
learning_rate = 0.0001
training_epochs = 15
batch_size = 100
keep_prob = tf.placeholder("float")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def layer_variable(x,W,b):
    L = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    L = tf.nn.relu(L+b)
    L = tf.nn.max_pool(L,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    L = tf.nn.dropout(L, keep_prob)
    return L

# layer 1 : Convolution
W1 = weight_variable([3,3,1,32])
b1 = bias_variable([32])
L1 = layer_variable(X_img, W1, b1) # shape : 14x14x32
w1_hist = tf.summary.histogram("weight1",W1)
b1_hist = tf.summary.histogram("bias1",b1)
l1_hist = tf.summary.histogram("layer1",L1)

# layer 2 : Convolution
W2 = weight_variable([3,3,32,64])
b2 = bias_variable([64])
L2 = layer_variable(L1, W2, b2) # shape : 7x7x64
w2_hist = tf.summary.histogram("weight2",W2)
b2_hist = tf.summary.histogram("bias2",b2)
l2_hist = tf.summary.histogram("layer2",L2)

# layer 3 : Convolution
W3 = weight_variable([3,3,64,128])
b3 = bias_variable([128])
L3 = layer_variable(L2, W3, b3)
L3 = tf.reshape(L3, [-1,128 * 4 * 4])
w3_hist = tf.summary.histogram("weight3",W3)
b3_hist = tf.summary.histogram("bias3",b3)
l3_hist = tf.summary.histogram("layer3",L3)

# layer 4 : fully Connected
W4 = tf.get_variable("W4",shape=[128*4*4,625], initializer = tf.contrib.layers.xavier_initializer())
b4 = bias_variable([625])
L4 = tf.nn.relu(tf.matmul(L3,W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob)
w4_hist = tf.summary.histogram("weight4",W4)
b4_hist = tf.summary.histogram("bias4",b4)
l4_hist = tf.summary.histogram("layer4",L4)


# layer 5 : fully Connected
W5 = tf.get_variable("W5",shape=[625,10], initializer = tf.contrib.layers.xavier_initializer())
b5 = bias_variable([10])
hypothesis = tf.matmul(L4,W5) + b5
y_conv = tf.nn.softmax(hypothesis)
w5_hist = tf.summary.histogram("weight5",W5)
b5_hist = tf.summary.histogram("bias5",b5)
l5_hist = tf.summary.histogram("layer5",y_conv)

cross_entropy = -tf.reduce_sum(Y*tf.log(y_conv))
cost_sum = tf.summary.scalar("cost",cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

saver = tf.train.Saver()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./board/mnist",sess.graph)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:1.0}
        # train_accuracy = sess.run(accuracy, feed_dict)
        c, _, = sess.run([cross_entropy,optimizer], feed_dict)
        avg_cost += c / total_batch
        summary = sess.run(merged, feed_dict=feed_dict)
        writer.add_summary(summary,i)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =','{:.9f}'.format(avg_cost))

save_path = saver.save(sess,"model_5layer.ckpt")
print('Accuracy : ', sess.run(accuracy, feed_dict={X:mnist.test.images, Y: mnist.test.labels, keep_prob:1.0}))
