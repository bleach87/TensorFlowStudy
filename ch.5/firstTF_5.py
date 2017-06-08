#_*_ coding: utf-8 _*_
# 딥러닝 - 합성곱 신경망
# 5 다중 계층 신경망

# 합성곱 신경망(Convolution neural network, CNN, ConvNet)
# 입력 데이터로 거의 이미지를 받음 => 신경망을 효율적으로 구현할 수 있고 필요한 매개변수의 수를 줄일 수 있음
# ex.MNIST

import tensorflow as tf

# MNIST 데이터 로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)

# 텐서플로 placeholder 정의
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

# 입력 데이터를 원래 이미지의 구조로 재구성(reshape함수 사용) : 입력 데이터의 크기를 4D텐서로 바꿈
# 두번째, 세번째 차원은 이미지의 너비와 높이
# 마지막 차원은 이미지의 컬러 채널 => 흑백이므로 1
# 신경망의 입력 : 28x28 크기의 2차원 공간의 뉴런
x_image = tf.reshape(x, [-1,28,28,1])

# 합성곱 계층
# 주요 목적 : 테두리선색 등 이미지의 시각정 특징,특성을 감지하는 것
# 입력 계층과 연결된 은닉 계층에 의해 처리
# CNN은 입력데이터가 첫번째 은닉계층의 뉴런에 완전 연결되어 있지 않음
# 이미지의 픽셀 정보를 저장하는 입력 뉴런의 작은 일부 영역만이 첫번째 은닉계층의 한 뉴런과 연결됨
# 28x28크기의 전체 입력 계층을 훑고 지나가는 5x5크기의 윈도 => 윈도는 계층의 전체 뉴런을 슬라이딩하여 지나감(왼쪽부터 오른쪽으로 한 픽셀씩 이동, 전체를 커버할 때까지 이동함)
# 윈도의 각 위치마다 입력 데이터를 처리하기 위한 은닉 계층의 뉴런이 배정됨
# 따라서 24x24크기의 은닉 계층을 만들게 됨
# 스트라이드(stride) : 윈도가 한번에 얼만큼 움직일지를 결정하는 매개변수
# 패딩(padding) : 이미지에 채울 테두리의 크기를 지정하는 매개변수
# => 좋은 결과를 내기 위해 이미지 바깥으로 윈도가 넘어갈 수 있도록 하는데 이를 위해 0(또는 다른값)으로 이미지의 바깥 테두리의 크기를 채움.

# 입력 계층과 은닉 계층의 뉴런을 연결하기 위해 5x5 가중치 행렬 W, 편향 b 가 필요함
# CNN 특징 : 5x5 가중치 행렬 W, 편향 b를 은닉계층의 모든 뉴런이 공유함
# 예제에서는 24x24(576)개의 뉴런이 같은 W와 b를 사용함
# 이에 따라 완전 연결 신경망에 비해 상당한 양의 가중치 매개변수가 감소함
# 만약, 가중치 행렬 W를 공유하지 않는다면 5x5x24x24(14000)개가 필요함
# 공유 행렬 W와 편향 b를 CNN에서는 커널(kernel) 또는 필터(filter)라고 부름
# => 이런 필터는 고유한 특징을 찾는 데 사용됨. 이미지를 리터치하는 이미지 처리 프로그램에서 사용하는 것과 유사함
# 하나의 커널은 이미지에서 한 종류의 특징만을 감지함 => 감지하고 싶은 각 특징에 한 개씩 여러개의 커널을 사용하는 것이 좋음

# 풀링 계층(pooling layer)
# 보통 합성곱 계층 뒤에 따라오는 것이 일반적
# 합성곱 계층의 출력값을 단순히 압축하고, 합성곱 계층이 생산한 정보를 컴팩트한 버전으로 만들어줌
# 예제에서는 합성곱 계층의 2x2영역을 풀링을 사용하여 하나의 점으로 데이터를 압축시킴
# 맥스풀링 : 2x2영역에서 가장 큰 값을 선택해서 정보를 압축함
# 합성곱 계층은 여러 개의 커널로 이뤄져 있으므로 각각에 대해 따로 맥스 풀링은 적용함
# 일반적으로, 여러 개의 풀링계층과 합성곱계층이 있을 수 있음
# 24x24합성곱의 결과를 2x2영역으로 분할하면 12x12개의 조각에 해당하는 12x12크기의 맥스풀링계층이 만들어짐(2x2 스트라이드를 거침)
# 합성곱 계층과는 달리, 데이터가 슬라이딩 윈도에 의해 생성되는 것이 아니라 타일처럼 나뉘어 각각 만들어짐
# 맥스 풀링은 어떤 특징의 이미지의 여러 곳에 나타날 때, 특징의 정확한 위치보다는 다른 특징들과의 상대적 위치가 더 중요함

# 가중치 행렬 W와 편향 b와 연관된 두개의 함수
# 가중치는 임의잡음으로 초기화
# 편향은 작은 양수(0.1)를 갖도록 초기화
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 합성곱 계층과 풀링 계층을 구성하기 위해 여러 개의 매개변수를 정해야 함
# 각 차원의 방향으로의 스트라이드(슬라이딩 윈도가 한번에 이동하는 크기)를 1로 하고 패딩은 'SAME'으로 지정
# 풀링은 2x2 크기의 맥스 풀링
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 첫 번째 합성곱 계층과 이를 뒤따르는 풀링계층 생성
# 예제에서는 윈도 크기가 5x5인 32개의 필터를 사용
# => 구조가 [5,5,1,32]인 가중치 행렬 W를 저장할 텐서를 정의
# 처음 두개의 차원은 윈도 크기, 세번째는 컬러 채널(흑백이므로 1), 마지막 차원은 얼마나 많은 특징(필터)을 사용할 것인지 정의
# 32개의 가중치 행렬에 대한 편향 정의
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# 렐루(ReLU) 활성화 함수는 최근 심층 신경망의 은닉 계층에서 거의 기본적으로 사용되는 활성화 함수
# 이 함수는 max(0,x)를 리턴
# 음수의 경우 0을 리턴, 그 외에는 x를 리턴
# 입력 이미지 x_image에 대해 합성곱을 적용하고 합성곱의 결과를 2D 텐서 W_conv1에 리턴
# 여기에 편향을 더해 최종적으로 렐루 활성화 합수를 적용함
# 다음으로 출력 값을 구하기 위해 맥스 풀링을 적용함
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 심층 신경망을 구성할 때는 여러 계층을 쌓아 올릴 수 있음
# 예시에서는 5x5 윈도에 64개의 필터를 갖는 두번째 합성곱 계층을 만듬
# 이 때는 이전 계층의 출력 값의 크기(32)를 채널의 수로 넘겨야 함
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 14x14 크기의 행렬인 h_pool1에 스트라이드 1로 5x5윈도를 적용하여 합성곱 계층을 만들었고, 맥스풀링까지 거쳐 크기가 7x7이 됨
# 마지막 소프트맥스 계층에 주입하기 위해 7x7출력값을 완전 연결 계층에 연결함
# 전체 이미지를 처리하기 위해 1024개의 뉴런을 사용
# 이에 따른 가중치와 편향 텐서
# 첫 번째 차원은 두번째 합성곱 계층의 7x7크기의 64개 필터
# 두 번째 차원은 임의로 선택한 뉴런의 개수(여기서는 1024)
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 텐서를 벡터로 변환
# 소프트맥스 함수는 이미지를 직렬화해서 벡터 형태로 입력함
# 이를 위해 가중치 행렬 W_fc1과 일차원 벡터를 곱하고 편향 b_fc1을 더한 후 렐루 활성화 함수를 적용함
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 드롭아웃 : 중도탈락 의미
# 신경망에서 필요한 매개변수 수를 줄이는 것
# 노드를 삭제하여 입력과 출력 사이의 연결을 제거하는것
# 어떤 뉴런을 제거하고 어떤 것을 유지할 지는 무작위로 결정됨
# 뉴런이 제거되거나 그렇지 않을 확률은 코드로 처리하지 않고 텐서플로에 위임함
# 드롭아웃은 모델이 데이터에 오버피팅(과적합)되는 것을 막아줌
# 오버피팅 : 은닉 계층에 아주 많은 수의 뉴런을 사용한다면 매우 상세한 모델을 만들 수 있으나, 동시에 임의의 잡음(오차)도 모델에 포함될 수 있음을 의미
# 입력 데이터의 차원에 비해 더 많은 매개변수를 가지는 모델에서 자주 일어나는 현상
# 오버피팅은 예측의 성능을 떨어뜨리므로 피하는 것이 좋음
# tf.nn.dropout함수를 사용하여 드롭아웃을 적용함

# 뉴런이 드롭아웃되지 않을 확률을 저장할 플레이스홀더 생성
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 모델에 소프트맥스 계층을 추가
# 소프트맥스 함수는 입력 이미지가 각 클래스에 속할 확률을 리턴하며 이 확률의 전체 합은 1이 됨
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 경사하강법 최적화 알고리즘을 ADAM 최적화 알고리즘으로 바꿔서 구현
# 드롭아웃 계층의 확률을 조절하는 추가 매개변수 keep_prob도 feed_dict 인수를 통해 전달
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    print("test accuracy %g"% sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
