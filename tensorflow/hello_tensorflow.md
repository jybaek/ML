# TensorFlow 간단한 실습
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
