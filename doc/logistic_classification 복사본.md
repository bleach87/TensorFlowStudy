<head>
    <script type="text/x-mathjax-config">
       MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
    </script>
    <script type="text/javascript" 
            src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
</head>

# Logistic(regression) Classification

#### Binary Classification
- 둘 중 하나를 고르는 Classification
    + ex1 : true,false 등 0과 1로 분류할 수 있어야함
- 0~1사이를 만들어 줄 수 있는 함수를 사용(Sigmoid 함수)
$$g(x) = \frac{1}{1+e^z}$$

- *Logistic Hypothesis*  
    + 기존 Hypothesis
        * $z=WX$
    + 새로운 Hypothesis
        * $H(x)=g(z)$
        * $H(x) = \frac{1}{1+e^-(W^TX)}$
---
#### Cost Function
- Cost Function By Linear Regression
    - $cost(W,b)=\frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)}-y^{(i)})^2$
    - 출력값이 항상 0~1사이로 만듬
        + $H(x) = W_x + b$ => $H(x)=\frac{1}{1+e^(-W^T)X}$ 

- New Cost Function For Logistic
    + Cost Function 은 예측한 값이 답에 가까우면 함수의 값이 작아짐
    + 반대로 예측한 값이 답과 멀어지면 함수의 값이 커지며, 이를 Hypothesis에 반영
    + 이에 따라 Classification을 위해서는 Log함수를 통해 도출해냄
        1. $cost(W) = \frac{1}{m}\sum C(H(x),y)$
        2. $C(H(x),y) = \begin{cases} -log(H(x)) & :y=1 \cr -log(1-H(x)) & :y=0 \end{cases}$
        3. 2의 수식을 tensorflow로 표현하기 어려워 하나의 식으로 표현하면, $C(H(x),y) = ylog(H(x))-(1-y)log(1-H(x))$
- Minimize Cost - Gradient Decsnt Algorithm
    - cost function  
        - $C(H(x),y) = ylog(H(x))-(1-y)log(1-H(x))$
        - `cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis)))`
    - minimize
        + $W := W - \alpha\frac{\partial}{\partial W}cost(W)$
            - `a = tf.Variable(0.1) # Learning rate, alpha`
            - `optimizer = tf.train.GradientDescentOptimizer(a)`
            - `train = optimizer.minimize(cost)`
---
#### Logistic Regression
$$H(x) = \frac{1}{1+e^-(W^TX)}$$
$$cost(W) = -\frac{1}{m}\sum ylog(H(x))-(1-y)log(1-H(x))$$
$$W := W - \alpha\frac{\partial}{\partial W}cost(W)$$

###### Source
    x_data = [[1, 2],[2, 3],[3, 1],[4, 3],[5, 3],[6, 2]]
    y_data = [[0],[0],[0],[1],[1],[1]]

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    # Weight : X가 2개가 들어오고 Y가 1개이기 때문에 1개가 나가서 [2,1]
    # bias : 나가는 값의 개수와 같으므로 [1]
    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

- sigmoid함수 사용 : $H(x) = \frac{1}{1+e^-(W^TX)}$  


    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

- cost : $cost(W) = -\frac{1}{m}\sum ylog(H(x))-(1-y)log(1-H(x))$


    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))

- $W := W - \alpha\frac{\partial}{\partial W}cost(W)$


    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    # Launch graph
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                print(step, cost_val)

        # Accuracy report
        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

