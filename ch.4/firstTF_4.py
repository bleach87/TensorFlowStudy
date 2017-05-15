#_*_ coding: utf-8 _*_
# MNIST
# 4 단일계층신경망

# tensorflow 내부의 학습데이터 가져오기
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

import tensorflow as tf
# 본 데이터는 배열 형태의 객체이므로 텐서플로의 convert_to_tensor함수를 이용해 텐서로 변환
# get_shape 함수로 구조 확인
# (55000,784)
# 첫번째 차원은 각 이미지에 대한 인덱스, 두번째 차원은 이미지안의 픽셀수(20x28)
# print tf.convert_to_tensor(mnist.train.images).get_shape()

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder("float", [None, 784])
# x = tf.placeholder(tf.float32, shape=[None,784])

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None, 10])
# y_ = tf.placeholder(tf.float32, shape=[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
print correct_prediction

print tf.argmax(y,1)

print tf.cast(correct_prediction, "float")

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
