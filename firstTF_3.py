#_*_ coding: utf-8 _*_
# 군집화 3
# 3.3 K-평균 알고리즘


# 샘플 데이터 생성
import numpy as np

num_points = 2000
vectors_set = []

for i in xrange(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0,0.9), np.random.normal(0.0,0.9)])
    else:
        vectors_set.append([np.random.normal(3.0,0.5), np.random.normal(1.0,0.5)])

import matplotlib.pyplot as plt
# 데이터 조작 패키지
import pandas as pd
# 시각화 패키지
import seaborn as sns

#난수 데이터 그래프
df = pd.DataFrame({"x":[v[0] for v in vectors_set], "y":[v[1] for v in vectors_set]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()

# 4개의 군집으로 그룹화하는 K-평균 알고리즘
import tensorflow as tf

# 무작위 데이터를 가지고 상수 텐서를 생성
vectors = tf.constant(vectors_set)
# 입력데이터에서 무작위로 K개의 데이터를 선택하는 방법 => 텐서플로가 무작위로 섞어서 K개의 중심을 선택하게 함
# K개의 데이터 포인트는 2D텐서로 저장됨
k=4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

# 텐서 구조 확인
print vectors.get_shape()
print centroids.get_shape()

# 각 점에 대해 유클리드 제곱거리를 구해 가장 가까운 중심을 계산한
# 유클리드 제곱거리는 그 자체로 거리를 나타내는 값은 아니고 여러 거리 사이의 대소를 비교할 때만 사용됨
# vectors,centroids 의 텐서가 2차원이긴 하지만 1차원의 크기가 다름(vectors:2000, centroids:4)
# 이 문제를 해결하기 위해 expand_dims 함수를 사용하여 두 텐서의 차원을 추가함 => 두 텐서를 2차원에서 3차원으로 만들어 뺄셈을 할 수 있도록 크기를 맞추기 위함
# vectors에 D0, centroids에 D1을 추가, D2=2 로 같음
# tensorShape에서 추가한 차원은 Dimension(1), 크기가 1로 확장된 차원이지만 텐서플로의 브로드캐스팅기능으로 tf.sub함수는 두 텐서의 각 원소를 어떻게 뺄지 알아냄
# 크기가 1인 차원은 텐서 연산시 다른 텐서의 해당 차원의 크기에 맞게 계산을 반복적으로 수행하면서 마치 차원이 늘어난 효과를 가짐
expanded_vectors = tf.expand_dims(vectors,0)
expanded_centroids = tf.expand_dims(centroids,1)

# 유클리드 제곱거리
# assignments를 나눠쓴 코드
# expanded_vectors,expanded_centroids에 대해 뺄셈을 한 결과(D0차원에는 중심(4개), D1차원에는 데이터의 인덱스(2000개), D2차원에는 x,y)
# v1.0 : sub -> subtract
# diff = tf.subtract(expanded_vectors, expanded_centroids)
# - diff의 제곱
# sqr = tf.square(diff)
# - 텐서의 차원을 감소시킴
# distance = tf.reduce_sum(sqr)
# - 지정된 차원에서 가장 작은 값의 인덱스를 리턴하는 argmin을 통해 각 데이터의 중심이 assignmanets에 할당됨
# assignments = tf.argmin(distance)
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroids)),2),0)

# 알고리즘에서 매 반복마다 새롭게 그룹화하면서 각 그룹에 해당하는 새로운 중심을 다시 계산함
# equal 함수를 사용하여 한 군집과 매칭되는(c에 매핑함) assignments텐서의 각 원소 위치를 True로 표시하는 Boolean Tensor(Dimension(2000))를 만듬
# where 함수를 사용하여 매개변수로 받은 Boolean Tensor에서 True로 표시된 위치를 값으로 가지는 텐서(Dimension(2000)*Dimension(1))를 만듬
# reshape 함수를 사용하여 c군집에 속한 vectors텐서의 포인트들의 인덱스로 구성된 텐서(Dimension(1)*Dimension(2000))을 만듬 => 텐서의 크기를 지정하는 매개변수의 두번째 배열원소가 -1이라 바로 앞 단계에서 만든 텐서의 차원을 뒤집는 효과를 가져옴
# gather 함수를 사용하여 c군집을 이루는 점들의 좌표를 모은 텐서(Dimension(1)*Dimension(2000)*Dimension(2))를 만듬
# reduce_mean 함수를 사용하여 c군집에 속한 모든 점의 평균값을 가진 텐서(Dimension(1)*Dimension(2))를 만듬
means = tf.concat([
    tf.reduce_mean(
        tf.gather(vectors,
                  tf.reshape(
                      tf.where(
                          tf.equal(assignments,c)
                      ),[1,-1])
                  ), reduction_indices=[1])
    for c in xrange(k)],0)
print means

# means텐서의 값을 centroids에 할당하는 연산 => run이 실행될 때 업데이트된 중심값이 다음번 루프에서 사용될 수 있음
update_centroids = tf.assign(centroids, means)
# 데이터 그래프를 실행하기 전에 모든 변수를 초기화함
# 1.0 -> tf.global_variables_initializer()
init_op = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init_op)

# 매 반복마다 중심은 업데이트 되고, 각 점은 새롭게 군집에 할당됨
for step in xrange(100):
    _, centroids_values, assignment_values = sess.run([update_centroids, centroids, assignments])

print "centroids"
print centroids_values

data = {"x": [],"y": [],"cluster": []}

for i in xrange(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()
