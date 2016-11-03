# 학습준비
### 학습 환경 구축
학습을 진행하는 리눅스는 CentOS 7.2 기준입니다. 기본적으로 python 2.7이 설치되어 있는데 python3을 사용하는 것이 좋아보입니다. 이유는 TensorFlow 관련해서 찾아보면 대부분의 문서나 실습이 python3 기준이기 때문입니다. 일단은 기본(python2)으로 진행하고 차후에 필요하면 업데이트 하도록 하겠습니다.

#### 필요한 도구 설치

```bash
$ yum install -y epel-release
$ yum install -y python-pip python-dev python-virtualenv
$ pip install jupyter
```


#### TensorFlow 설치

```bash
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Install from sources" below.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc2-cp27-none-linux_x86_64.whl
$ pip install --upgrade $TF_BINARY_URL
$ source ~/tensorflow/bin/activate
```
