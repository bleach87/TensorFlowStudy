#_*_ coding: utf-8 _*_
# 선형회귀분석 2.1

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 학습데이터 생성
num_points = 1000
vectors_set = []

for i in xrange(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# 그래프로 보여주는 부분
# plt.plot(x_data, y_data, 'ro')
# plt.show()

# 매개변수 지정(가설지정)
W = tf.Variable(tf.random_uniform([1],-1.0,-1.0))
b = tf.Variable(tf.zeros([1]))
y = x_data * W + b

# 경사 하강법에 의한 비용함수(오차함수) => 함수값을 최소화하는 알고리즘
# y값과 y_data값의 차이를 알기위해 그 사이의 거리를 제곱한 후 평균내어 계산함
loss = tf.reduce_mean(tf.square(y - y_data))

# 경사하강법은 초기 시작점에서 함수의 값이 최소화되는 방향으로 매개변수를 변경하는 것을 반복적으로 수행하는 알고리즘
# W(기울기)를 음의 방향쪽으로 진행하면서 반복적으로 최적화 수행
# 보통 양의 값을 만들기위해 거리값을 제곱, 기울기 계산을 위해 오차함수는 미분이 가능해야 함
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 세션 초기화
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 세션 반복수행(훈련)
# 경사 하강법 알고지즘을 수행할 때마다 평면의 한 지점에서 시작하여 더 작은 오차를 갖는 직선을 찾아 이동
# 오차함수의 기울기를 계산하기 위해 텐서플로는 오차함수를 미분함
# 반복이 일어날 때마다 움직힐 방향을 알아내기 위해 W,b에 대한 편미분방정식 계산이 필요함
for step in xrange(8):
    sess.run(train)
    # print step, sess.run(W),sess.run(b)
    plt.plot(x_data,y_data,'ro')
    plt.plot(x_data,sess.run(W)*x_data * sess.run(b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print(step,sess.run(loss))

