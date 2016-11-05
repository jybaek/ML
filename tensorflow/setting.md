# 학습준비
### 학습 환경 구축
학습을 진행하는 리눅스는 CentOS 7.2 기준입니다. 기본적으로 python 2.7이 설치되어 있는데 python3을 사용하는 것이 좋아보입니다. 이유는 TensorFlow 관련해서 찾아보면 대부분의 문서나 실습이 python3 기준이기 때문입니다. 일단은 기본(python2)으로 진행하고 차후에 필요하면 업데이트 하도록 하겠습니다.

그리고 TensorFlow를 설치하고 사용하는 방법에는 [여러가지](https://github.com/tensorflowkorea/tensorflow-kr/blob/master/g3doc/get_started/os_setup.md)가 있지만 여기서는 [Pip](https://github.com/tensorflowkorea/tensorflow-kr/blob/master/g3doc/get_started/os_setup.md#pip-installation)를 사용합니다.

#### 필요한 도구 설치
```bash
$ yum install -y epel-release
$ yum install -y python-pip python-dev
$ pip install jupyter
```
prompt에서 바로 실습해도 상관없고 .py로 실습해도 상관없지만 여기서는 jupyter를 사용합니다. 

#### TensorFlow 바이너리 선택
```bash
# Ubuntu/Linux 64-bit, GPU 버전, Python 2.7 (CentOS도 상관 없음)
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
```

#### TensorFlow 설치
```bash
# Python 2
sudo pip install --upgrade $TF_BINARY_URL
```

#### TensorFlow 실행
```bash
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```
