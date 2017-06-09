# -*- coding: utf-8 -*-
# Reference : niektemme/tensorflow-mnist-predict
#


import sys
import tensorflow as tf
from PIL import Image, ImageFilter


def predictint(imvalue):

    """
    훈련된 모델을 사용하여 하나의 단일 이미지를 받아서 예측한다
    :param imvalue:
    :return:
    """

    """
    훈련시와 동일한 모델을 다시 정의합니다.
    자세한 설명은 훈력쪽 소크 코드에서 확인할 수 있습니다.
    """
    # x = tf.placeholder(tf.float32, [None, 784])
    x = tf.placeholder("float", shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    """
    모델을 saver 를 사용하여 복구합니다.
    sess.run(init_op)
    saver.restore(sess, "model.ckpt")
    """
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model.ckpt")
        prediction = tf.argmax(y_conv, 1)
        return sess.run(prediction, feed_dict={x: [imvalue], keep_prob: 1.0})


def imageprepare(argv):
    """
    로컬에서 이미지를 받아서 Tensorflow 처리 가능한 형태로 변환하는 역할을 수행합니다.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # 우리가 테스트할 네트워크는 28/28 이미지이다

    # 입력된 28/28이 아닌 이미지를 28/28로 변환하기 위해 가로 세로 중 어느쪽이 큰지 확인
    if width > height:
        # 폭이 더 큰 경우 처리 로직
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width

        # 20/20 이미지로 변환하고
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  #
        newImage.paste(img, (4, wtop))  # 리사이즈된 이미지를 흰색 바탕의 캔버스에 붙여 넣는다
    else:
        # 높이가 더 큰경우에 처리 로직
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # 픽셀 데이터로 변환

    # 255의 RGB 0 흰색, 1 검은색의 이진수로 노멀라이제이션 작업을 수행
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


def main():
    """
    Main function.
    """

    argv = 'number5.png'
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    print (predint[0])  # first value in list


if __name__ == "__main__":
    main()