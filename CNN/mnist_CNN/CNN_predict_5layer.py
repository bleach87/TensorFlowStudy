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
    X = tf.placeholder("float", shape=[None, 784])
    Y = tf.placeholder("float", shape=[None, 10])
    X_img = tf.reshape(X, [-1, 28, 28, 1])
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

    def layer_variable(x, W, b):
        L = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        L = tf.nn.relu(L + b)
        L = tf.nn.max_pool(L, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L = tf.nn.dropout(L, keep_prob)
        return L

    # layer 1 : Convolution
    W1 = weight_variable([3, 3, 1, 32])
    b1 = bias_variable([32])
    L1 = layer_variable(X_img, W1, b1)  # shape : 14x14x32

    # layer 2 : Convolution
    W2 = weight_variable([3, 3, 32, 64])
    b2 = bias_variable([64])
    L2 = layer_variable(L1, W2, b2)  # shape : 7x7x64

    # layer 3 : Convolution
    W3 = weight_variable([3, 3, 64, 128])
    b3 = bias_variable([128])
    L3 = layer_variable(L2, W3, b3)
    L3 = tf.reshape(L3, [-1, 128 * 4 * 4])

    # layer 4 : fully Connected
    W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
    b4 = bias_variable([625])
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob)

    # layer 5 : fully Connected
    W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
    b5 = bias_variable([10])
    hypothesis = tf.matmul(L4, W5) + b5
    y_conv = tf.nn.softmax(hypothesis)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    """
    모델을 saver 를 사용하여 복구합니다.
    sess.run(init_op)
    saver.restore(sess, "model.ckpt")
    """
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model_5layer.ckpt")
        prediction = tf.argmax(y_conv, 1)
        return sess.run(prediction, feed_dict={X: [imvalue], keep_prob: 1.0})


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