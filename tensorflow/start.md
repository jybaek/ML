# TensorFlow ?
## 인공지능? 머신러닝? 딥러닝? 

### 소개
이쪽 분야를 시작하게 되면 가장 먼저 헷갈리는 것이 인공지능, 머신러닝, 딥러닝에 대한 단어의 차이입니다. 간략하게 정리하는 차원으로 기록합니다.

인공지능이라는 단어의 어원을 따라가 보면 놀랍게도 1950년대부터 사용됐다는 사실을 알 수 있습니다. 그리고 그 탄생은 [1956년에 이르러서, 학문 분야로 들어섰습니다](https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5#.EC.9D.B8.EA.B3.B5.EC.A7.80.EB.8A.A5.EC.9D.98_.ED.83.84.EC.83.9D.281952-1956.29). 그 후 데이터 패턴을 학습하여 새로운 데이터에 대해 예측하거나 처리하는 머신러닝은 1980년대에, 인간의 뇌에 있는 신경망을 모델로 제작된 뉴럴 네트워크와 관련있는 딥러닝(deep learning)은 2010년쯤 관심받기 시작합니다. 

멀게만 느껴졌던 분야가 일반인에게 밀접하게 다가오는 계기가 있었으니 너무도 유명한 이세돌 9단과 알파고의 바둑 대결이었습니다. 그러다 보니 비슷비슷한 단어들에 대해 많은 질문을 받기도 합니다. 

자, 정리하면 인공지능, 머신러닝, 딥러닝은 아래와 같은 집합입니다.

![jpeg](https://tensorflowkorea.files.wordpress.com/2016/08/ai-ml-dl.jpg?w=359&h=305)
> 출처:[tensorflowkorea](https://tensorflowkorea.wordpress.com/%ED%95%B4%EC%BB%A4%EC%97%90%EA%B2%8C-%EC%A0%84%ED%95%B4%EB%93%A4%EC%9D%80-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-1/)

### 딥러닝 종류 
머신러닝과 딥러닝에 대한 이야기는 인터넷에 너무나 많으므로 굳이 설명할 필요가 없습니다. 또한 그 기술을 다루는 라이브러리는 상당히 다양한데 여기서는 TensorFlow를 다루도록 합니다. 굳이 TensorFlow인 이유는 github에서 가장 핫한 머신러닝 프로젝트이기도 하고.. 역시 구글 이기 때문입니다. 

![jpeg](https://tensorflowkorea.files.wordpress.com/2016/05/comparison_of_deep_learning_package.png)
> 출처:[tensorflowkorea](https://tensorflowkorea.wordpress.com/2016/05/21/%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%8C%A8%ED%82%A4%EC%A7%80-%EB%B9%84%EA%B5%90/)

후에 나머지도 천천히 실펴보도록 합니다. 

### TensorFlow 간단한 실습
실습은 python prompt를 이용해도 되고, vim을 사용해도 됩니다. 하지만 여기서는 jupyter notebook을 사용했습니다. 일단 앞서 [학습준비](tensorflow/setting.md#tensorflow-실행) 과정에서 사용된 코드를 다시 실습해봅시다.

```python
>>> import tensorflow as tf
```
위 문장의 의미는 C나 C++ 개념으로 살펴보면 tensorflow라고 하는 library를 가져오는 개념입니다. include보다는 -l을 사용해서 library에 있는 함수나 정의를 사용할 수 있도록 선언했다고 보면 맞을 것 같습니다. 뒤에 as는 앞으로 그 library를 사용할 때 tf라는 별칭을 사용하겠다는 의미입니다.

```python
>>> hello = tf.constant('Hello, TensorFlow!')
```
tensorflow는 여러 가지 메소드를 갖고있고 앞으로 그 메소드를 사용하며 다양한 실습을 진행할텐데 그 중 타입 선언을 하는 구문 중 하나인 [constant](https://www.tensorflow.org/versions/r0.11/api_docs/python/constant_op.html#constant) 입니다. 일반적인 C/C++의 const 처럼 한번 선언되면 변경할 수 없다는 특징을 갖습니다. 이렇게 선언된 constant를 hello에 담았습니다.


```python
>>> sess = tf.Session()
```
tensorflow에서 [Session](https://www.tensorflow.org/versions/r0.11/api_docs/python/client.html#Session)이라고 하는 클래스입니다. 오퍼레이션이 실행되고 평가되는 환경을 만들어 줍니다. python으로 작성하는 TensorFlow 코드를 실제 CPU나 GPU에 의해 동작할 수 있도록 연결해주는 고리라고 생각하면 됩니다. 모든 코드는 항상 Session 클래스를 생성하게 됩니다.

```python
>>> print(sess.run(hello))
```
C/C++의 printf와 같은 역할을 수행하는 print입니다. hello constant를 실제로 실행합니다. 여기서 sess.run()의 결과를 print 하는 이유는 print(hello)의 경우 hello에 대한 설명만 출력되기 때문(억지로 빗대자면 객체의 내용을 출력해야 하는데 주소를 출력하는 형태)인데, 연산 결과를 얻기 위해서는 실제 연산이 이루어지는 Session을 통해야 합니다. 

```python
>>> print(hello)
Tensor("Const:0", shape=(), dtype=string)
>>> 
```
위처럼 print(hello)의 결과는 Tensor에 대한 설명만 출력됩니다.


```python
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```
두 개의 constant를 생성하고 그 합을 tensorflow를 통해 연산합니다. 위의 설명처럼 단순히 print(a+b)를 실행할 경우는 Tensor에 대한 출력만을 보이게 됩니다. 항상 Session을 통해 CPU에 명령을 전달해야 한다는 사실을 잊지 않도록 합니다.