# 단일 계층 신경망: MNIST

딥 러닝의 대표적인 패턴 인식으로 예전부터 손글씨 숫자 이미지를 분류하는 것은 인공지능 분야의 기초였습니다. 텐서플로우에서도 예외 없이 이를 실습할 수 있습니다. 학습을 위한 데이터셋은 텐서플로우가 설치될 때 기본적으로 built-in되어 있으므로 별도로 다운로드가 필요 없습니다.
### [[텐서플로 첫걸음]](https://tensorflowkorea.wordpress.com/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EC%B2%AB%EA%B1%B8%EC%9D%8C/) 4.1 MNIST 데이터셋

```python
# MNIST 데이터셋을 가져옵니다.
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# output :
# Extracting MNIST_data/train-images-idx3-ubyte.gz
# Extracting MNIST_data/train-labels-idx1-ubyte.gz
# Extracting MNIST_data/t10k-images-idx3-ubyte.gz
# Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

import tensorflow as tf

# 배열 형태의 객체를 텐서 형태로 변환하고 구조를 확인합니다.
tf.convert_to_tensor(mnist.train.images).get_shape()

# output : TensorShape([Dimension(55000), Dimension(784)])
# 이미지는 총 55,000개가 있으며 각 이미지는 784개의 필셀로 되어 있습니다.

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", [None, 784])

# 모델을 생성합시다.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 크로스 엔트로피 함수를 구현하는 과정입니다.
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 학습 속도 0.01과 경사 하강법 알고리즘을 사용하여 크로스 엔트로피를 최소화하는 역전파 알고리즘을 사용합니다.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 텐서플로우 연산을 실행할 수 있도록 세션을 생성합니다.
sess=tf.Session()

# 모든 변수를 초기화하고 세션을 시작합니다.
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 모델을 평가하도록 합니다.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 위 코드는 불리언으로 이루어진 리스트를 리턴합니다. 예측한 것이 얼만큼 맞았는지 불리언을 수치 값(부동소수점)으로 변경합니다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 정확도를 계산해봅시다.
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# output : 0.9103
```
